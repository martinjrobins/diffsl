use std::collections::HashMap;

use log::{debug, log_enabled, Level};
use ndarray::s;

use crate::ast::{self, Ast, AstKind, StringSpan};

use super::{
    can_broadcast_to, layout::ArcLayout, Layout, LayoutKind, Shape, Tensor, TensorBlock,
    ValidationError, ValidationErrors,
};

pub struct EnvVar {
    layout: ArcLayout,
    is_time_dependent: bool,
    is_state_dependent: bool,
    is_dstatedt_dependent: bool,
    is_input_dependent: bool,
    is_algebraic: bool,
}

impl EnvVar {
    pub fn is_time_dependent(&self) -> bool {
        self.is_time_dependent
    }

    pub fn is_state_dependent(&self) -> bool {
        self.is_state_dependent
    }

    pub fn is_dstatedt_dependent(&self) -> bool {
        self.is_dstatedt_dependent
    }

    pub fn is_algebraic(&self) -> bool {
        self.is_algebraic
    }

    pub fn is_input_dependent(&self) -> bool {
        self.is_input_dependent
    }

    pub fn layout(&self) -> &Layout {
        self.layout.as_ref()
    }
}

pub struct Env {
    current_span: Option<StringSpan>,
    errs: ValidationErrors,
    vars: HashMap<String, EnvVar>,
    inputs: Vec<String>,
}

impl Env {
    pub fn new(inputs: &[&str]) -> Self {
        let mut vars = HashMap::new();
        vars.insert(
            "t".to_string(),
            EnvVar {
                layout: ArcLayout::new(Layout::new_scalar()),
                is_time_dependent: true,
                is_state_dependent: false,
                is_dstatedt_dependent: false,
                is_input_dependent: false,
                is_algebraic: true,
            },
        );
        Env {
            errs: ValidationErrors::default(),
            vars,
            current_span: None,
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// create a new ArcLayout from a Layout, if the layout already exists in the env then return the existing one
    pub fn new_layout_ptr(&mut self, layout: Layout) -> ArcLayout {
        for var in self.vars.values() {
            if var.layout.as_ref() == &layout {
                return var.layout.clone();
            }
        }
        ArcLayout::new(layout)
    }

    pub fn is_tensor_time_dependent(&self, tensor: &Tensor) -> bool {
        if tensor.name() == "u" || tensor.name() == "dudt" {
            return true;
        };
        tensor.elmts().iter().any(|block| {
            block
                .expr()
                .get_dependents()
                .iter()
                .any(|&dep| dep == "t" || self.vars[dep].is_time_dependent())
        })
    }
    pub fn is_tensor_state_dependent(&self, tensor: &Tensor) -> bool {
        self.is_tensor_dependent_on(tensor, "u")
    }

    pub fn is_tensor_input_dependent(&self, tensor: &Tensor) -> bool {
        self.inputs
            .iter()
            .any(|input| self.is_tensor_dependent_on(tensor, input))
    }

    pub fn is_tensor_dstatedt_dependent(&self, tensor: &Tensor) -> bool {
        self.is_tensor_dependent_on(tensor, "dudt")
    }

    fn is_tensor_dependent_on(&self, tensor: &Tensor, var: &str) -> bool {
        if tensor.name() == var {
            return true;
        };
        tensor.elmts().iter().any(|block| {
            block.expr().get_dependents().iter().any(|&dep| {
                dep == var
                    || match var {
                        "u" => self.vars[dep].is_state_dependent(),
                        "dudt" => self.vars[dep].is_dstatedt_dependent(),
                        // must be an input
                        _ => self.vars[dep].is_input_dependent(),
                    }
            })
        })
    }

    pub fn push_var(&mut self, var: &Tensor) {
        self.vars.insert(
            var.name().to_string(),
            EnvVar {
                layout: var.layout_ptr().clone(),
                is_algebraic: true,
                is_time_dependent: self.is_tensor_time_dependent(var),
                is_state_dependent: self.is_tensor_state_dependent(var),
                is_dstatedt_dependent: self.is_tensor_dstatedt_dependent(var),
                is_input_dependent: self.is_tensor_input_dependent(var),
            },
        );
    }

    pub fn push_var_blk(&mut self, var: &Tensor, var_blk: &TensorBlock) {
        self.vars.insert(
            var_blk.name().unwrap().to_string(),
            EnvVar {
                layout: var_blk.layout_ptr().clone(),
                is_algebraic: true,
                is_time_dependent: self.is_tensor_time_dependent(var),
                is_state_dependent: self.is_tensor_state_dependent(var),
                is_dstatedt_dependent: self.is_tensor_dstatedt_dependent(var),
                is_input_dependent: self.is_tensor_input_dependent(var),
            },
        );
    }

    pub fn get(&self, name: &str) -> Option<&EnvVar> {
        self.vars.get(name)
    }

    fn get_layout_binary_op<'s>(
        &mut self,
        left: &Ast<'s>,
        right: &Ast<'s>,
        op: &ast::Binop,
        indices: &Vec<char>,
    ) -> Option<Layout> {
        let left_layout = self.get_layout(left, indices)?;
        let right_layout = self.get_layout(right, indices)?;
        match Layout::broadcast(vec![left_layout, right_layout], Some(op.op)) {
            Ok(layout) => Some(layout),
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}. Op is {}, lhs is {}, rhs is {}.", e, op.op, left, right),
                    left.span,
                ));
                None
            }
        }
    }

    fn get_layout_name(
        &mut self,
        name: &str,
        ast: &Ast,
        rhs_indices: &[char],
        lhs_indices: &[char],
        indice: Option<&Ast>,
    ) -> Option<Layout> {
        let var = self.get(name);
        if var.is_none() {
            self.errs.push(ValidationError::new(
                format!("cannot find variable {name}"),
                ast.span,
            ));
            return None;
        }
        let var = var.unwrap();
        let layout = var.layout();

        if rhs_indices.len() < layout.min_rank() {
            self.errs.push(ValidationError::new(
                format!(
                    "cannot index variable {} with {} indices. Expected at least {} indices",
                    name,
                    rhs_indices.len(),
                    layout.rank()
                ),
                ast.span,
            ));
            return None;
        }
        let mut permutation = vec![0; rhs_indices.len()];
        for i in 0..rhs_indices.len() {
            permutation[i] = match lhs_indices.iter().position(|&x| x == rhs_indices[i]) {
                Some(pos) => pos,
                None => {
                    // if we are indexing a single element then we can allow for missing indices
                    let mut allow_missing = false;
                    if let Some(indice) = indice {
                        let indice = indice.kind.as_indice().unwrap();
                        if indice.sep.is_none() || indice.last.is_none() {
                            allow_missing = true;
                        }
                    };
                    if !allow_missing {
                        self.errs.push(ValidationError::new(
                            format!(
                                "cannot find index {} in lhs indices {:?} ",
                                rhs_indices[i], lhs_indices
                            ),
                            ast.span,
                        ));
                        return None;
                    }
                    0
                }
            }
        }
        let layout_permuted = match layout.permute(permutation.as_slice()) {
            Ok(layout) => layout,
            Err(e) => {
                self.errs
                    .push(ValidationError::new(format!("{e}"), ast.span));
                return None;
            }
        };

        if let Some(indice) = indice {
            let indice = indice.kind.as_indice().unwrap();
            // we'll only support indexing dense 1D variables for now
            // a dense 1d variable could be a column vector with shape (n, 1) or a row vector with shape (n)
            // or an nd array with shape (1, 1, ..., n, 1)
            let is_one_d = layout_permuted.shape().iter().filter(|&&d| d != 1).count() == 1;
            if !is_one_d || layout_permuted.kind() != &LayoutKind::Dense {
                self.errs.push(ValidationError::new(
                    format!(
                        "can only index dense 1D variables. Variable {} has layout {}",
                        name, layout_permuted
                    ),
                    ast.span,
                ));
                return None;
            }

            // a separator without a last value is an error
            if indice.sep.is_some() && indice.last.is_none() {
                self.errs.push(ValidationError::new(
                    "range indice must have an end value".to_string(),
                    ast.span,
                ));
                return None;
            }

            // if the indice is a single integer then the resulting layout is a scalar
            if indice.sep.is_none() {
                return Some(Layout::new_scalar());
            } else {
                // if the indice is a range then the resulting layout is a dense layout with shape given by the range
                // along the only non-unit dimension of the variable
                let first = indice.first.kind.as_integer().unwrap();
                let last = indice.last.as_ref().unwrap().kind.as_integer().unwrap();
                // make sure the range is valid
                if last < first {
                    self.errs.push(ValidationError::new(
                        format!(
                            "invalid range indice: start {} is greater than end {}",
                            first, last
                        ),
                        ast.span,
                    ));
                    return None;
                }
                let dim = usize::try_from(last - first).unwrap();
                let shape = layout_permuted
                    .shape()
                    .map(|&d| if d != 1 { dim } else { 1 });
                return Some(Layout::new_dense(Shape::from(shape)));
            }
        }

        Some(layout_permuted)
    }

    fn get_layout_call(
        &mut self,
        call: &ast::Call,
        ast: &Ast,
        indices: &Vec<char>,
    ) -> Option<Layout> {
        let layouts = call
            .args
            .iter()
            .map(|c| self.get_layout(c, indices))
            .collect::<Option<Vec<Layout>>>()?;
        match Layout::broadcast(layouts, None) {
            Ok(layout) => Some(layout),
            Err(e) => {
                self.errs
                    .push(ValidationError::new(format!("{e}"), ast.span));
                None
            }
        }
    }

    pub fn get_layout(&mut self, ast: &Ast, indices: &Vec<char>) -> Option<Layout> {
        let layout = match &ast.kind {
            AstKind::Assignment(a) => self.get_layout(a.expr.as_ref(), indices),
            AstKind::Binop(binop) => {
                self.get_layout_binary_op(binop.left.as_ref(), binop.right.as_ref(), binop, indices)
            }
            AstKind::Monop(monop) => self.get_layout(monop.child.as_ref(), indices),
            AstKind::Call(call) => self.get_layout_call(call, ast, indices),
            AstKind::CallArg(arg) => self.get_layout(arg.expression.as_ref(), indices),
            AstKind::Number(_) => Some(Layout::new_scalar()),
            AstKind::Integer(_) => Some(Layout::new_scalar()),
            AstKind::Domain(d) => Some(Layout::new_dense(Shape::zeros(1) + d.dim)),
            AstKind::Name(name) => self.get_layout_name(
                name.name,
                ast,
                &name.indices,
                indices,
                name.indice.as_ref().map(|i| i.as_ref()),
            ),
            _ => panic!("unrecognised ast node {:#?}", ast.kind),
        };
        if log_enabled!(Level::Debug) {
            let indices_str = layout.as_ref().map(|l| {
                l.explicit_indices()
                    .iter()
                    .map(|i| {
                        i.into_iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>()
                            .join(", ")
                    })
                    .collect::<Vec<String>>()
            });
            debug!(
                "layout for ast {} with indices {:?} is {} with indices {:?}",
                ast,
                indices,
                layout.as_ref().unwrap_or(&Layout::new_scalar()),
                indices_str.unwrap_or_default()
            );
        }
        layout
    }

    // returns a tuple of (expr_layout, elmt_layout) giving the layouts of the expression and the tensor element.)
    pub fn get_layout_tensor_elmt(
        &mut self,
        elmt: &ast::TensorElmt,
        indices: &[char],
        force_dense: bool,
    ) -> Option<(Layout, Layout)> {
        let expr_indices = elmt.expr.get_indices();
        // get any indices from the expression that do not appear in 'indices' and add them to 'indices' to a new vector
        let mut new_indices = indices.to_vec();
        for i in expr_indices {
            if !indices.contains(&i) && !new_indices.contains(&i) {
                new_indices.push(i);
            }
        }

        // TODO: for now we will only support contractions from 2d to 1d
        if new_indices.len() > indices.len() && (new_indices.len() != 2 || indices.len() != 1) {
            self.errs.push(ValidationError::new(
                format!(
                    "contraction only supported from 2D to 1D tensors. Got {}D to {}D",
                    new_indices.len(),
                    indices.len()
                ),
                elmt.expr.span,
            ));
            return None;
        }
        debug!(
            "calculating expr layout for tensor element with expr: {}",
            elmt.expr
        );
        let mut expr_layout = self.get_layout(elmt.expr.as_ref(), &new_indices)?;
        if force_dense {
            expr_layout.to_dense();
        }

        // broadcast the expression layout to the tensor rank
        // (tensor rank given by the number of indices)
        // if we have an additional index then we contract the last dimension of the expression layout to get the final layout
        let expr_layout_to_rank = if new_indices.len() > indices.len() {
            match expr_layout.contract_last_axis() {
                Ok(layout) => layout,
                Err(e) => {
                    self.errs
                        .push(ValidationError::new(format!("{e}"), elmt.expr.span));
                    return None;
                }
            }
        } else {
            expr_layout.broadcast_to_rank(indices.len())
        };

        // calculate the shape of the tensor element.
        let elmt_layout = if elmt.indices.is_none() {
            // If there are no indicies then the layout is the same as the expression layout
            expr_layout_to_rank
        } else {
            // If there are indicies then the rank is determined by the number of indices, and the
            // shape is determined by the ranges of the indices
            // TODO: this is quite large, perhaps move to another function

            // make sure the number of indices matches the number of dimensions
            let elmt_indices = elmt.indices.as_ref().unwrap();
            let given_indices_ast = &elmt_indices.kind.as_vector().unwrap().data;
            let given_indices: Vec<&ast::Indice> = given_indices_ast
                .iter()
                .map(|i| i.kind.as_indice().unwrap())
                .collect();
            if given_indices.len() != indices.len() {
                self.errs.push(ValidationError::new(
                    format!(
                        "number of dimensions of tensor element ({}) does not match number of dimensions of tensor ({})",
                        given_indices.len(), indices.len()
                    ),
                    elmt_indices.span,
                ));
                return None;
            }

            let mut exp_expr_shape = Shape::ones(indices.len());

            // we will use the expression shape as defaults if the range is not explicitly given
            exp_expr_shape
                .slice_mut(s![..expr_layout_to_rank.rank()])
                .assign(expr_layout_to_rank.shape());

            // calculate the shape of the tensor element from the given indices and expression shape
            let all_range_indices = given_indices.iter().all(|i| i.sep == Some(".."));
            let mut old_dim = None;
            for (i, indice) in given_indices.iter().enumerate() {
                let first = indice.first.kind.as_integer().unwrap();

                // make sure the use of the range separator is valid
                if !all_range_indices && matches!(indice.sep, Some("..")) {
                    self.errs.push(ValidationError::new(
                        "can only use range separator if all indices are ranges".to_string(),
                        given_indices_ast[i].span,
                    ));
                }
                let dim = if indice.sep.is_some() {
                    if let Some(second) = &indice.last {
                        let second = second.kind.as_integer().unwrap();
                        if second < first {
                            self.errs.push(ValidationError::new(
                                "range end must be greater than range start".to_string(),
                                given_indices_ast[i].span,
                            ));
                            return None;
                        }
                        usize::try_from(second - first).unwrap()
                    } else {
                        exp_expr_shape[i]
                    }
                } else {
                    1usize
                };

                // make sure the dimension of the range is consistent
                if all_range_indices && old_dim.is_some() && dim != old_dim.unwrap() {
                    self.errs.push(ValidationError::new(
                        "range indices must have the same dimension".to_string(),
                        given_indices_ast[i].span,
                    ));
                    return None;
                }
                old_dim = Some(dim);
                exp_expr_shape[i] = dim;
            }

            // check that the expression shape can be broadcast to the tensor element shape
            if !can_broadcast_to(&exp_expr_shape, expr_layout_to_rank.shape()) {
                self.errs.push(ValidationError::new(
                    format!(
                        "cannot broadcast expression shape {} to tensor element shape {}",
                        expr_layout_to_rank.shape(),
                        exp_expr_shape
                    ),
                    elmt.expr.span,
                ));
                return None;
            }

            // tensor elmt layout is:
            // 1. dense if the expression is dense and no indices are ranges
            // 2. diagonal if the expression is dense and all indices are ranges, or the expression is diagonal and no indices are ranges
            // 3. sparse if the expression is sparse and no indices are blocks
            let elmt_layout = match expr_layout.kind() {
                LayoutKind::Dense => {
                    if all_range_indices {
                        Layout::new_diagonal(exp_expr_shape)
                    } else {
                        Layout::new_dense(exp_expr_shape)
                    }
                }
                LayoutKind::Sparse => {
                    if all_range_indices {
                        self.errs.push(ValidationError::new(
                            "cannot use range indices with sparse expression".to_string(),
                            elmt.expr.span,
                        ));
                        return None;
                    } else {
                        expr_layout.broadcast_to_shape(&exp_expr_shape)
                    }
                }
                LayoutKind::Diagonal => {
                    if all_range_indices {
                        self.errs.push(ValidationError::new(
                            "cannot use range indices with diagonal expression".to_string(),
                            elmt.expr.span,
                        ));
                        return None;
                    } else {
                        Layout::new_diagonal(exp_expr_shape)
                    }
                }
            };
            elmt_layout
        };

        Some((expr_layout, elmt_layout))
    }

    pub fn current_span(&self) -> Option<StringSpan> {
        self.current_span
    }

    pub fn set_current_span(&mut self, current_span: Option<StringSpan>) {
        self.current_span = current_span;
    }

    pub fn errs(&self) -> &ValidationErrors {
        &self.errs
    }

    pub fn errs_mut(&mut self) -> &mut ValidationErrors {
        &mut self.errs
    }
}
