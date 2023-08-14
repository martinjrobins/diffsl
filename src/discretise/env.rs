use std::collections::HashMap;

use ndarray::s;

use crate::ast::{StringSpan, Ast, self, AstKind};

use super::{layout::RcLayout, ValidationErrors, Layout, Tensor, TensorBlock, ValidationError, Shape, can_broadcast_to, LayoutKind};

pub struct EnvVar {
    layout: RcLayout,
    is_time_dependent: bool,
    is_state_dependent: bool,
    is_algebraic: bool,
}

impl EnvVar {
    pub fn is_time_dependent(&self) -> bool {
        self.is_time_dependent
    }

    pub fn is_state_dependent(&self) -> bool {
        self.is_state_dependent
    }

    pub fn is_algebraic(&self) -> bool {
        self.is_algebraic
    }

    pub fn layout(&self) -> &Layout {
        self.layout.as_ref()
    }
}

pub struct Env {
    current_span: Option<StringSpan>,
    errs: ValidationErrors,
    vars: HashMap<String, EnvVar>,
}

impl Env {
    pub fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert(
            "t".to_string(),
            EnvVar {
                layout: RcLayout::new(Layout::new_scalar()),
                is_time_dependent: true,
                is_state_dependent: false,
                is_algebraic: true,
            },
        );
        Env {
            errs: ValidationErrors::new(),
            vars,
            current_span: None,
        }
    }
    pub fn is_tensor_time_dependent(&self, tensor: &Tensor) -> bool {
        if tensor.name() == "u" || tensor.name() == "dudt" { return true };
        tensor.elmts().iter().any(|block| {
            block
                .expr()
                .get_dependents()
                .iter()
                .any(|&dep| dep == "t" || self.vars[dep].is_time_dependent())
        })
    }
    pub fn is_tensor_state_dependent(&self, tensor: &Tensor) -> bool {
        if tensor.name() == "u" || tensor.name() == "dudt" { return true };
        tensor.elmts().iter().any(|block| {
            block
                .expr()
                .get_dependents()
                .iter()
                .any(|&dep| dep == "u" || self.vars[dep].is_state_dependent())
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
            },
        );
    }

    pub fn push_var_blk(&mut self, var: &Tensor, var_blk: &TensorBlock) {
        self.vars.insert(
            var_blk.name().unwrap().to_string(),
            EnvVar {
                layout: var_blk.layout().clone(),
                is_algebraic: true,
                is_time_dependent: self.is_tensor_time_dependent(var),
                is_state_dependent: self.is_tensor_state_dependent(var),
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
        match Layout::broadcast(vec![left_layout, right_layout], op.op == '*') {
            Ok(layout) => Some(layout),
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}", e),
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
        rhs_indices: &Vec<char>,
        lhs_indices: &Vec<char>,
    ) -> Option<Layout> {
        let var = self.get(name);
        if var.is_none() {
            self.errs.push(ValidationError::new(
                format!("cannot find variable {}", name),
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
                    self.errs.push(ValidationError::new(
                        format!("cannot find index {} in lhs indices {:?} ", rhs_indices[i], lhs_indices),
                        ast.span,
                    ));
                    return None;
                }
            }
        }
        let layout_permuted = match layout.permute(permutation.as_slice()) {
            Ok(layout) => layout,
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}", e),
                    ast.span,
                ));
                return None;
            }
        };
                
        Some(layout_permuted)
    }

    

    fn get_layout_call(&mut self, call: &ast::Call, ast: &Ast, indices: &Vec<char>) -> Option<Layout> {
        let layouts = call
            .args
            .iter()
            .map(|c| self.get_layout(c, indices))
            .collect::<Option<Vec<Layout>>>()?;
        match Layout::broadcast(layouts, false) {
            Ok(layout) => Some(layout),
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}", e),
                    ast.span,
                ));
                None
            }
        }    
    }

    pub fn get_layout(&mut self, ast: &Ast, indices: &Vec<char>) -> Option<Layout> {
        match &ast.kind {
            AstKind::Assignment(a) => self.get_layout(a.expr.as_ref(), indices),
            AstKind::Binop(binop) => {
                self.get_layout_binary_op(binop.left.as_ref(), binop.right.as_ref(), binop, indices)
            }
            AstKind::Monop(monop) => self.get_layout(monop.child.as_ref(), indices),
            AstKind::Call(call) => self.get_layout_call(&call, ast, indices),
            AstKind::CallArg(arg) => self.get_layout(arg.expression.as_ref(), indices),
            AstKind::Number(_) => Some(Layout::new_scalar()),
            AstKind::Integer(_) => Some(Layout::new_scalar()),
            AstKind::Domain(d) => Some(Layout::new_dense(Shape::zeros(1) + d.dim)),
            AstKind::IndexedName(name) => {
                self.get_layout_name(name.name, ast, &name.indices, indices)
            }
            AstKind::Name(name) => self.get_layout_name(name, ast, &vec![], indices),
            _ => panic!("unrecognised ast node {:#?}", ast.kind),
        }
    }
    

    // returns a tuple of (expr_layout, elmt_layout) giving the layouts of the expression and the tensor element.)
    pub fn get_layout_tensor_elmt(&mut self, elmt: &ast::TensorElmt, indices: &[char]) -> Option<(Layout, Layout)> {
        let expr_indices = elmt.expr.get_indices();
        // get any indices from the expression that do not appear in 'indices' and add them to 'indices' to a new vector
        let mut new_indices = indices.to_vec();
        for i in expr_indices {
            if !indices.contains(&i) && !new_indices.contains(&i) {
                new_indices.push(i);
            }
        }
        
        // TODO: for now we will only support one additional index
        if new_indices.len() > indices.len() + 1 {
            self.errs.push(ValidationError::new(
                format!(
                    "cannot index tensor element with more than one additional index. Found {} indices",
                    new_indices.len() - indices.len()
                ),
                elmt.expr.span,
            ));
            return None;
        }

        let expr_layout = self.get_layout(elmt.expr.as_ref(), &new_indices)?;
        
        // calculate the shape of the tensor element. 
        let elmt_layout = if elmt.indices.is_none() {
            // If there are no indices then blk layout is the same as the expression, but broadcast to the tensor rank
            // (tensor rank given by the number of indices)
            // if we have an additional index then we contract the last dimension of the expression layout to get the final layout
            if new_indices.len() > indices.len() {
                match expr_layout.contract_last_axis() {
                    Ok(layout) => layout,
                    Err(e) => {
                        self.errs.push(ValidationError::new(
                            format!("{}", e),
                            elmt.expr.span,
                        ));
                        return None;
                    }
                }
            } else {
                expr_layout.to_rank(indices.len()).unwrap()
            }
        } else {
            // If there are indicies then the rank is determined by the number of indices, and the
            // shape is determined by the ranges of the indices
            // TODO: this is quite large, perhaps move to another function
            

            // make sure the number of indices matches the number of dimensions
            let elmt_indices = elmt.indices.as_ref().unwrap();
            let given_indices_ast = &elmt_indices.kind.as_vector().unwrap().data;
            let given_indices: Vec<&ast::Indice> = given_indices_ast.iter().map(|i| i.kind.as_indice().unwrap()).collect();
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
            exp_expr_shape.slice_mut(s![..expr_layout.rank()]).assign(&expr_layout.shape());
            
            // calculate the shape of the tensor element from the given indices and expression shape
            let all_range_indices = given_indices.iter().all(|i| i.sep == Some(".."));
            let mut old_dim = None;
            for (i, indice) in given_indices.iter().enumerate() {
                let first = indice.first.kind.as_integer().unwrap();
                
                // make sure the use of the range separator is valid
                if !all_range_indices && matches!(indice.sep, Some("..")) {
                    self.errs.push(ValidationError::new(
                        format!("can only use range separator if all indices are ranges"),
                        given_indices_ast[i].span,
                    ));
                }
                let dim = if let Some(_) = indice.sep {
                    if let Some(second) = &indice.last {
                        let second = second.kind.as_integer().unwrap();
                        if second < first {
                            self.errs.push(ValidationError::new(
                                format!("range end must be greater than range start"),
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
                        format!("range indices must have the same dimension"),
                        given_indices_ast[i].span,
                    ));
                    return None;
                }
                old_dim = Some(dim);
                exp_expr_shape[i] = dim;
            }

            // check that the expression shape can be broadcast to the tensor element shape
            if !can_broadcast_to(&exp_expr_shape, &expr_layout.shape()) {
                self.errs.push(ValidationError::new(
                    format!("cannot broadcast expression shape {} to tensor element shape {}", expr_layout.shape(), exp_expr_shape),
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
                },
                LayoutKind::Sparse => {
                    if all_range_indices {
                        self.errs.push(ValidationError::new(
                            format!("cannot use range indices with sparse expression"),
                            elmt.expr.span,
                        ));
                        return None;
                    } else {
                        expr_layout.clone()    
                    }
                },
                LayoutKind::Diagonal => {
                    if all_range_indices {
                        self.errs.push(ValidationError::new(
                            format!("cannot use range indices with diagonal expression"),
                            elmt.expr.span,
                        ));
                        return None;
                    } else {
                        Layout::new_diagonal(exp_expr_shape)
                    }
                },
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