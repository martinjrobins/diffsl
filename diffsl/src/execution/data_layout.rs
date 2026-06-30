use std::cmp::min;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use log::debug;

use crate::{
    ast::{Ast, AstKind, Name, StringSpan},
    discretise::{
        ArcLayout, DiscreteModel, Index, Layout, Tensor, TensorBlock, ValidationError,
        ValidationErrors,
    },
};

use super::{Translation, TranslationFrom};

// there are three different layouts:
// 1. the data layout is a mapping from tensors to the index of the first element in the data or constants array.
//    Each tensor in the data or constants layout is a contiguous array of nnz elements
// 2. the layout layout is a mapping from Layout to the index of the first element in the indices array.
//    Only sparse layouts are stored, and each sparse layout is a contiguous array of nnz*rank elements
// 3. the translation layout is a mapping from layout from-to pairs to the index of the first element in the indices array.
//    Each contraction pair is an array of nnz-from elements, each representing the indices of the "to" tensor that will be summed into.
// We also store a mapping from tensor names to their layout, so that we can easily look up the layout of a tensor

/// Metadata for a single interp1d call, populated after constant evaluation.
#[derive(Debug, Clone)]
pub struct Interp1dInfo {
    pub x_offset: usize,
    pub y_offset: usize,
    pub n: usize,
    /// If the x values are uniformly spaced, this is the increment.
    pub dx: Option<f64>,
    /// Hash of the Call AST for fallback lookup when span is unavailable.
    pub hash_key: u64,
}

#[derive(Debug)]
pub struct DataLayout {
    is_constant_map: HashMap<String, bool>,
    data_index_map: HashMap<String, usize>,
    data_length_map: HashMap<String, usize>,
    layout_index_map: HashMap<ArcLayout, usize>,
    binary_layout_index_map: HashMap<(ArcLayout, ArcLayout, Vec<usize>), usize>,
    translate_index_map: HashMap<(ArcLayout, ArcLayout), usize>,
    data: Vec<f64>,
    constants: Vec<f64>,
    indices: Vec<i32>,
    layout_map: HashMap<String, ArcLayout>,
    named_data_index_map: HashMap<String, usize>,
    interp1d_map: HashMap<StringSpan, Interp1dInfo>,
    interp1d_hash_map: HashMap<u64, Interp1dInfo>,
}

impl DataLayout {
    pub fn new(model: &DiscreteModel) -> Result<Self, ValidationErrors> {
        let mut is_constant_map = HashMap::new();
        let mut data_index_map = HashMap::new();
        let mut data_length_map = HashMap::new();
        let mut layout_index_map = HashMap::new();
        let mut translate_index_map = HashMap::new();
        let mut data = Vec::new();
        let mut constants = Vec::new();
        let mut indices = Vec::new();
        let mut layout_map = HashMap::new();
        let mut binary_layout_index_map = HashMap::new();
        let mut named_data_index_map = HashMap::new();

        // add layout info for "t"
        let t_layout = ArcLayout::new(Layout::new_scalar());
        layout_map.insert("t".to_string(), t_layout);
        is_constant_map.insert("t".to_string(), false);

        // add layout info for model index "N"
        let n_layout = ArcLayout::new(Layout::new_scalar());
        layout_map.insert("N".to_string(), n_layout);
        is_constant_map.insert("N".to_string(), false);

        let mut add_tensor = |tensor: &Tensor, in_data: bool, in_constants: bool| {
            // insert the data (non-zeros) for each tensor
            layout_map.insert(tensor.name().to_string(), tensor.layout_ptr().clone());
            if in_data {
                data_index_map.insert(tensor.name().to_string(), data.len());
                data_length_map.insert(tensor.name().to_string(), tensor.nnz());
                debug!(
                    "adding tensor {} to data at index {} with nnz {}",
                    tensor.name(),
                    data.len(),
                    tensor.nnz()
                );
                data.extend(vec![0.0; tensor.nnz()]);
                is_constant_map.insert(tensor.name().to_string(), false);
            } else if in_constants {
                data_index_map.insert(tensor.name().to_string(), constants.len());
                debug!(
                    "adding tensor {} to constants at index {} with nnz {}",
                    tensor.name(),
                    constants.len(),
                    tensor.nnz()
                );
                data_length_map.insert(tensor.name().to_string(), tensor.nnz());
                constants.extend(vec![0.0; tensor.nnz()]);
            }
            is_constant_map.insert(tensor.name().to_string(), in_constants);

            // add the translation info for each block-tensor pair
            for blk in tensor.elmts() {
                // need layouts and is_constant of all named tensor blocks
                if let Some(name) = blk.name() {
                    layout_map.insert(name.to_string(), blk.layout().clone());
                    is_constant_map.insert(name.to_string(), in_constants);
                    if in_data || in_constants {
                        let mut block_start = Index::zeros(tensor.rank());
                        for (axis, start) in blk.start().iter().enumerate().take(tensor.rank()) {
                            block_start[axis] = *start;
                        }
                        let block_offset = tensor.layout().find_nnz_index(&block_start).unwrap();
                        named_data_index_map.insert(
                            name.to_string(),
                            data_index_map[tensor.name()] + block_offset,
                        );
                    }
                }

                // insert the layout info for each tensor expression
                if !layout_index_map.contains_key(blk.expr_layout()) {
                    layout_index_map.insert(blk.expr_layout().clone(), indices.len());
                    let data_layout = blk.expr_layout().to_data_layout();
                    debug!(
                        "adding layout for block {} in tensor {} at index {}: {:?}",
                        blk.name().unwrap_or("<unnamed>"),
                        tensor.name(),
                        indices.len(),
                        data_layout
                    );
                    indices.extend(blk.expr_layout().to_data_layout());
                }

                // if any tensors in the block expression have a different layout to the block expression
                // then we need to add a binary layout translation
                for (tensor_name, tensor_indices) in blk.expr().get_dependents_with_indices() {
                    let tensor_layout = layout_map.get(tensor_name).unwrap();
                    if tensor_layout != blk.expr_layout() {
                        let permutation = Self::permutation(blk, &tensor_indices, tensor_layout);
                        if !binary_layout_index_map.contains_key(&(
                            tensor_layout.clone(),
                            blk.expr_layout().clone(),
                            permutation.clone(),
                        )) {
                            let blayout = tensor_layout
                                .to_binary_data_layout(blk.expr_layout(), &permutation);
                            if !blayout.is_empty() {
                                debug!(
                                    "adding binary layout from {} to {} with permutation {:?}: {:?}",
                                    tensor_name,
                                    blk.name().unwrap_or(tensor.name()),
                                    permutation,
                                    blayout
                                );
                                binary_layout_index_map.insert(
                                    (
                                        tensor_layout.clone(),
                                        blk.expr_layout().clone(),
                                        permutation,
                                    ),
                                    indices.len(),
                                );
                                indices.extend(blayout);
                            }
                        }
                    }
                }

                // and the translation info for each block-tensor pair
                if let std::collections::hash_map::Entry::Vacant(e) =
                    translate_index_map.entry((blk.expr_layout().clone(), blk.layout().clone()))
                {
                    let translation = Translation::new(
                        blk.expr_layout(),
                        blk.layout(),
                        blk.start(),
                        tensor.layout_ptr(),
                    );
                    debug!(
                        "adding translation from {} to {}: {:?}",
                        blk.name().unwrap_or("<unnamed>"),
                        tensor.name(),
                        translation
                    );
                    e.insert(indices.len());
                    indices.extend(translation.to_data_layout());
                }
            }
        };

        model
            .constant_defns()
            .iter()
            .for_each(|c| add_tensor(c, false, true));
        if let Some(input) = model.input() {
            add_tensor(input, true, false);
        }
        model
            .input_dep_defns()
            .iter()
            .for_each(|i| add_tensor(i, true, false));

        model
            .time_dep_defns()
            .iter()
            .for_each(|i| add_tensor(i, true, false));

        add_tensor(model.state(), false, false);
        if let Some(state_dot) = model.state_dot() {
            add_tensor(state_dot, false, false);
        }
        model
            .state_dep_defns()
            .iter()
            .for_each(|i| add_tensor(i, true, false));
        model
            .state_dep_post_f_defns()
            .iter()
            .for_each(|i| add_tensor(i, true, false));
        if let Some(lhs) = model.lhs() {
            add_tensor(lhs, false, false);
        }
        add_tensor(model.rhs(), false, false);
        if let Some(out) = model.out() {
            add_tensor(out, false, false);
        }

        let mut ret = Self {
            is_constant_map,
            data_index_map,
            layout_index_map,
            data,
            indices,
            translate_index_map,
            layout_map,
            data_length_map,
            constants,
            binary_layout_index_map,
            named_data_index_map,
            interp1d_map: HashMap::new(),
            interp1d_hash_map: HashMap::new(),
        };

        for tensor in model.constant_defns() {
            for blk in tensor.elmts() {
                ret.evaluate_constant_block(tensor, blk);
            }
        }

        let errors = ret.verify_interp1d_sortedness(model);
        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(ret)
    }

    fn find_interp1d_calls<'a>(
        expr: &'a Ast,
        f: &mut impl FnMut(&'a crate::ast::Call<'a>, Option<StringSpan>),
    ) {
        match &expr.kind {
            AstKind::Call(call) => {
                if call.fn_name == "interp1d" {
                    f(call, expr.span);
                }
                for arg in &call.args {
                    Self::find_interp1d_calls(arg, f);
                }
            }
            AstKind::CallArg(arg) => Self::find_interp1d_calls(&arg.expression, f),
            AstKind::Binop(binop) => {
                Self::find_interp1d_calls(&binop.left, f);
                Self::find_interp1d_calls(&binop.right, f);
            }
            AstKind::Monop(monop) => Self::find_interp1d_calls(&monop.child, f),
            AstKind::Assignment(a) => Self::find_interp1d_calls(&a.expr, f),
            _ => {}
        }
    }

    fn evaluate_interp1d_arg(&self, arg: &Ast) -> Vec<f64> {
        // determine the length n from the expression's dependent constant tensors
        let deps = arg.get_dependents();
        let n = deps
            .iter()
            .find_map(|name| self.data_length_map.get(*name).copied())
            .unwrap_or_else(|| panic!("interp1d: cannot determine length of arg"));

        // clone the arg so we can own it in a TensorBlock
        let expr = arg.clone();
        let blk = TensorBlock::new_dense_vector(None, 0, n, expr);

        blk.expr_layout()
            .indices()
            .enumerate()
            .map(|(expr_index, index)| {
                self.evaluate_constant_expr(blk.expr(), &blk, expr_index, &index)
            })
            .collect()
    }

    fn verify_interp1d_sortedness(&mut self, model: &DiscreteModel) -> ValidationErrors {
        let mut errors = ValidationErrors::default();

        for tensor in model.all_tensors() {
            for blk in tensor.elmts() {
                Self::find_interp1d_calls(blk.expr(), &mut |call, span| {
                    let x_vals = self.evaluate_interp1d_arg(&call.args[0]);
                    let y_vals = self.evaluate_interp1d_arg(&call.args[1]);

                    if x_vals.len() != y_vals.len() {
                        errors.push(ValidationError::new(
                            "interp1d: x and y must have the same length".to_string(),
                            span,
                        ));
                        return;
                    }
                    if let Some(w) = x_vals.windows(2).find(|w| w[0] >= w[1]) {
                        errors.push(ValidationError::new(
                            format!(
                                "interp1d: x must be strictly increasing, found non-increasing pair ({}, {})",
                                w[0], w[1]
                            ),
                            span,
                        ));
                        return;
                    }

                    let dx = if x_vals.len() >= 2 {
                        let d = x_vals[1] - x_vals[0];
                        let tol = 1e-12 * d.abs().max(1.0);
                        if x_vals.windows(2).all(|w| (w[1] - w[0] - d).abs() < tol) {
                            Some(d)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let n = x_vals.len();
                    let x_offset = self.constants.len();
                    self.constants.extend_from_slice(&x_vals);
                    let y_offset = self.constants.len();
                    self.constants.extend_from_slice(&y_vals);

                    // deterministic hash from the call args (x, y, q)
                    // — must match codegen; only hash base args, never the full Call
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    format!("{:?}", &call.args[0]).hash(&mut hasher);
                    format!("{:?}", &call.args[1]).hash(&mut hasher);
                    format!("{:?}", &call.args[2]).hash(&mut hasher);
                    let hash_key = hasher.finish();

                    let info = Interp1dInfo {
                        x_offset,
                        y_offset,
                        n,
                        dx,
                        hash_key,
                    };
                    if let Some(s) = span {
                        self.interp1d_map.insert(s, info.clone());
                    }
                    self.interp1d_hash_map.insert(hash_key, info);
                });
            }
        }

        errors
    }

    pub fn get_interp1d_info(
        &self,
        span: Option<&StringSpan>,
        hash_key: u64,
    ) -> Option<&Interp1dInfo> {
        if let Some(s) = span {
            if let Some(info) = self.interp1d_map.get(s) {
                return Some(info);
            }
        }
        self.interp1d_hash_map.get(&hash_key)
    }

    fn evaluate_constant_block(&mut self, tensor: &Tensor, blk: &TensorBlock) {
        let expr_values = if blk.layout().values().is_some() {
            blk.layout().values().unwrap().to_vec()
        } else {
            blk.expr_layout()
                .indices()
                .enumerate()
                .map(|(expr_index, index)| {
                    self.evaluate_constant_expr(blk.expr(), blk, expr_index, &index)
                })
                .collect::<Vec<_>>()
        };
        let block_values =
            Self::translate_constant_values(blk.expr_layout(), blk.layout(), &expr_values);
        assert_eq!(
            block_values.len(),
            blk.layout().nnz(),
            "constant block {} evaluated to {} values, but block layout has {} non-zeros",
            blk.name().unwrap_or("<unnamed>"),
            block_values.len(),
            blk.layout().nnz()
        );

        let tensor_start = self.data_index_map[tensor.name()];
        for (relative_index, value) in blk.layout().indices().zip(block_values.iter().copied()) {
            let mut absolute_index = Index::zeros(tensor.rank());
            for axis in 0..min(relative_index.len(), absolute_index.len()) {
                absolute_index[axis] = relative_index[axis];
            }
            for (axis, start) in blk.start().iter().enumerate().take(absolute_index.len()) {
                absolute_index[axis] += *start;
            }
            let Some(offset) = tensor.layout().find_nnz_index(&absolute_index) else {
                panic!(
                    "constant block index {:?} not found in tensor {} layout",
                    absolute_index,
                    tensor.name()
                );
            };
            self.constants[tensor_start + offset] = value;
        }
    }

    fn translate_constant_values(
        source_layout: &ArcLayout,
        block_layout: &ArcLayout,
        values: &[f64],
    ) -> Vec<f64> {
        let translation = TranslationFrom::new(source_layout, block_layout);
        match translation {
            TranslationFrom::ElementWise => values.to_vec(),
            TranslationFrom::Broadcast {
                broadcast_by: _,
                broadcast_len,
            } => values
                .iter()
                .flat_map(|value| std::iter::repeat_n(*value, broadcast_len))
                .collect(),
            TranslationFrom::DenseContraction {
                contract_by: _,
                contract_len,
            } => values
                .chunks(contract_len)
                .map(|chunk| chunk.iter().sum())
                .collect(),
            TranslationFrom::DiagonalContraction { contract_by: _ } => values.to_vec(),
            TranslationFrom::SparseContraction {
                contract_by: _,
                contract_start_indices,
                contract_end_indices,
            } => contract_start_indices
                .iter()
                .zip(contract_end_indices.iter())
                .map(|(start, end)| values[*start..*end].iter().sum())
                .collect(),
        }
    }

    fn evaluate_constant_expr(
        &self,
        expr: &Ast,
        blk: &TensorBlock,
        expr_index: usize,
        index: &Index,
    ) -> f64 {
        match &expr.kind {
            AstKind::Assignment(assignment) => {
                self.evaluate_constant_expr(assignment.expr.as_ref(), blk, expr_index, index)
            }
            AstKind::Binop(binop) => {
                let left = self.evaluate_constant_expr(binop.left.as_ref(), blk, expr_index, index);
                let right =
                    self.evaluate_constant_expr(binop.right.as_ref(), blk, expr_index, index);
                match binop.op {
                    '+' => left + right,
                    '-' => left - right,
                    '*' => left * right,
                    '/' => left / right,
                    unknown => panic!("unknown constant binary op '{unknown}'"),
                }
            }
            AstKind::Monop(monop) => {
                let child =
                    self.evaluate_constant_expr(monop.child.as_ref(), blk, expr_index, index);
                match monop.op {
                    '+' => child,
                    '-' => -child,
                    unknown => panic!("unknown constant unary op '{unknown}'"),
                }
            }
            AstKind::Call(call) if call.fn_name == "interp1d" => {
                let q = self.evaluate_constant_expr(&call.args[2], blk, expr_index, index);
                let x_vals = self.evaluate_interp1d_arg(&call.args[0]);
                let y_vals = self.evaluate_interp1d_arg(&call.args[1]);
                let n = x_vals.len();
                let qc = q.clamp(x_vals[0], x_vals[n - 1]);
                let mut lo = 0usize;
                let mut hi = n - 1;
                while lo < hi {
                    let mid = (lo + hi).div_ceil(2);
                    if x_vals[mid] <= qc {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                let k = lo.min(n - 2);
                let t = (qc - x_vals[k]) / (x_vals[k + 1] - x_vals[k]);
                y_vals[k] + t * (y_vals[k + 1] - y_vals[k])
            }
            AstKind::Call(call) => {
                let args = call
                    .args
                    .iter()
                    .map(|arg| self.evaluate_constant_expr(arg.as_ref(), blk, expr_index, index))
                    .collect::<Vec<_>>();
                Self::evaluate_constant_call(call.fn_name, &args)
            }
            AstKind::CallArg(arg) => {
                self.evaluate_constant_expr(arg.expression.as_ref(), blk, expr_index, index)
            }
            AstKind::SparseImport(import) => {
                panic!(
                    "read('{}') constant import should be evaluated from layout values",
                    import.path
                )
            }
            AstKind::Name(name) => self.evaluate_constant_name(name, blk, expr_index, index),
            AstKind::Number(value) => *value,
            AstKind::Integer(value) => *value as f64,
            AstKind::NamedGradient(name) => {
                panic!("named gradient {name} is not a constant expression")
            }
            AstKind::Index(_) => panic!("index AST nodes are not supported in constant values"),
            AstKind::Slice(_) => panic!("slice AST nodes are not supported in constant values"),
            other => panic!("unexpected AST node in constant expression: {other:?}"),
        }
    }

    fn evaluate_constant_call(name: &str, args: &[f64]) -> f64 {
        match (name, args) {
            ("sin", [x]) => x.sin(),
            ("cos", [x]) => x.cos(),
            ("tan", [x]) => x.tan(),
            ("sinh", [x]) => x.sinh(),
            ("cosh", [x]) => x.cosh(),
            ("tanh", [x]) => x.tanh(),
            ("exp", [x]) => x.exp(),
            ("log", [x]) => x.ln(),
            ("log10", [x]) => x.log10(),
            ("sqrt", [x]) => x.sqrt(),
            ("abs", [x]) => x.abs(),
            ("sigmoid", [x]) => 1.0 / (1.0 + (-x).exp()),
            ("arcsinh", [x]) => x.asinh(),
            ("arccosh", [x]) => x.acosh(),
            ("heaviside", [x]) => {
                if *x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ("copysign", [x, y]) => x.copysign(*y),
            ("pow", [x, y]) => x.powf(*y),
            ("min", [x, y]) => x.min(*y),
            ("max", [x, y]) => x.max(*y),
            _ => panic!(
                "unknown constant function call '{}' with {} args",
                name,
                args.len()
            ),
        }
    }

    fn evaluate_constant_name(
        &self,
        name: &Name,
        blk: &TensorBlock,
        expr_index: usize,
        index: &Index,
    ) -> f64 {
        if name.name == "t" || name.name == "N" {
            panic!("{} is not a constant expression", name.name);
        }

        let Some(is_constant) = self.is_constant_map.get(name.name) else {
            panic!("cannot find variable {}", name.name);
        };
        if !is_constant {
            panic!("{} is not a constant expression", name.name);
        }
        if name.is_tangent {
            return 0.0;
        }

        let layout = self.layout_map.get(name.name).unwrap_or_else(|| {
            panic!("cannot find layout for constant variable {}", name.name);
        });
        let tensor_start = self
            .data_index_map
            .get(name.name)
            .copied()
            .or_else(|| self.named_data_index_map.get(name.name).copied())
            .unwrap_or_else(|| panic!("cannot find data for constant variable {}", name.name));
        let elmt_index = if layout.is_dense() {
            self.evaluate_dense_constant_name_index(name, blk, index, layout)
        } else if layout.is_sparse() || layout.is_diagonal() {
            if blk.expr_layout() != layout {
                let permutation = Self::permutation(blk, name.indices.as_slice(), layout);
                let binary_layout = layout.to_binary_data_layout(blk.expr_layout(), &permutation);
                if binary_layout.is_empty() {
                    Some(expr_index)
                } else {
                    let mapped_index = binary_layout[expr_index];
                    if mapped_index < 0 {
                        None
                    } else {
                        Some(usize::try_from(mapped_index).unwrap())
                    }
                }
            } else {
                Some(expr_index)
            }
        } else {
            panic!("unexpected layout for constant variable {}", name.name);
        };

        elmt_index.map_or(0.0, |offset| self.constants[tensor_start + offset])
    }

    fn evaluate_dense_constant_name_index(
        &self,
        name: &Name,
        blk: &TensorBlock,
        index: &Index,
        layout: &ArcLayout,
    ) -> Option<usize> {
        let mut name_index = Vec::new();
        for (axis, c) in name.indices.iter().enumerate() {
            let pi = blk
                .indices()
                .iter()
                .position(|idx| idx == c)
                .unwrap_or(blk.indices().len());
            let value = if let Some(indice_ast) = name.indice.as_ref() {
                let Some(indice) = indice_ast.kind.as_indice() else {
                    panic!("invalid index expression '{}'", indice_ast);
                };
                let start = Self::evaluate_constant_integer_expr(indice.first.as_ref());
                if indice.last.is_some() {
                    index.get(pi).copied().unwrap_or(0) + start
                } else {
                    start
                }
            } else {
                index.get(pi).copied().unwrap_or(0)
            };
            if layout.shape()[axis] == 1 {
                name_index.push(0);
            } else {
                name_index.push(value);
            }
        }

        for (axis, value) in name_index
            .iter()
            .enumerate()
            .take(min(name_index.len(), layout.rank()))
        {
            if *value < 0 || *value >= layout.shape()[axis] as i64 {
                return None;
            }
        }

        if name_index.is_empty() {
            Some(0)
        } else {
            Some(Layout::ravel_index(
                &Index::from_vec(name_index),
                layout.shape(),
            ))
        }
    }

    fn evaluate_constant_integer_expr(expr: &Ast) -> i64 {
        match &expr.kind {
            AstKind::Integer(value) => *value,
            AstKind::Number(value) => {
                if value.fract() != 0.0 {
                    panic!("non-integer value '{}' in integer expression", value);
                }
                *value as i64
            }
            AstKind::Name(name) => {
                if name.name == "N" {
                    panic!("N is not allowed in a constant integer expression");
                }
                panic!(
                    "unsupported name '{}' in constant integer expression",
                    name.name
                );
            }
            AstKind::Monop(monop) => {
                let child = Self::evaluate_constant_integer_expr(monop.child.as_ref());
                match monop.op {
                    '+' => child,
                    '-' => -child,
                    unknown => panic!("unknown integer unary op '{unknown}'"),
                }
            }
            AstKind::Binop(binop) => {
                let left = Self::evaluate_constant_integer_expr(binop.left.as_ref());
                let right = Self::evaluate_constant_integer_expr(binop.right.as_ref());
                match binop.op {
                    '+' => left + right,
                    '-' => left - right,
                    '*' => left * right,
                    '/' => left / right,
                    '%' => left % right,
                    unknown => panic!("unknown integer binary op '{unknown}'"),
                }
            }
            other => panic!("unexpected integer expression: {other:?}"),
        }
    }

    /// construct a permutation from the block expression indices to the tensor indices
    /// in case they are in a different order
    /// if any indices appear in the tensor indices but not in the block indices, we add these
    /// to the end of the permutation (these will be contracted indices)
    /// if any indices appear in the block indices but not in the tensor indices, we
    /// map them to the end (these will be broadcasted indices)
    ///
    /// case 1: no contraction, translate
    /// (i, j) -> (j, i) permutation [1, 0]
    /// case 2: contraction, translate, always contract last index
    /// (i) -> (j, i) permutation [1, 1]
    /// case 3: contraction with tranlation with broadcast
    /// (i) -> (j) permutation [1, 0]
    pub fn permutation(
        blk: &TensorBlock,
        tensor_indices: &[char],
        tensor_layout: &ArcLayout,
    ) -> Vec<usize> {
        let mut permutation = blk
            .indices()
            .iter()
            .map(|idx| {
                tensor_indices
                    .iter()
                    .position(|&c| c == *idx)
                    .unwrap_or(tensor_layout.rank())
            })
            .collect::<Vec<usize>>();
        for (i, index) in tensor_indices.iter().enumerate() {
            if !blk.indices().contains(index) {
                permutation.push(i);
            }
        }
        permutation
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&String, bool)> {
        self.data_index_map
            .keys()
            .map(|name| (name, *self.is_constant_map.get(name).unwrap()))
    }

    // get the layout of a tensor by name
    pub fn get_layout(&self, name: &str) -> Option<&ArcLayout> {
        self.layout_map.get(name)
    }

    pub fn is_constant(&self, name: &str) -> bool {
        *self.is_constant_map.get(name).unwrap()
    }

    // get the index of the data array for the given tensor name
    pub fn get_data_index(&self, name: &str) -> Option<usize> {
        self.data_index_map.get(name).copied()
    }

    pub fn format_data(&self, data: &[f64]) -> String {
        let mut data_index_sorted: Vec<_> = self.data_index_map.iter().collect();
        data_index_sorted.sort_by_key(|(_, index)| **index);
        let mut s = String::new();
        s += "[";
        for (name, index) in data_index_sorted {
            let nnz = self.data_length_map[name];
            s += &format!("{}: {:?}, ", name, &data[*index..*index + nnz]);
        }
        s += "]";
        s
    }

    pub fn get_tensor_data(&self, name: &str) -> Option<&[f64]> {
        let index = self.get_data_index(name)?;
        let nnz = self.get_data_length(name)?;
        Some(&self.data()[index..index + nnz])
    }
    pub fn get_tensor_constants(&self, name: &str) -> Option<&[f64]> {
        if !self.is_constant(name) {
            return None;
        }
        let index = self.get_data_index(name)?;
        let nnz = self.get_data_length(name)?;
        Some(&self.constants()[index..index + nnz])
    }
    pub fn get_tensor_data_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        let index = self.get_data_index(name)?;
        let nnz = self.get_data_length(name)?;
        Some(&mut self.data_mut()[index..index + nnz])
    }

    pub fn get_data_length(&self, name: &str) -> Option<usize> {
        self.data_length_map.get(name).copied()
    }

    pub fn get_layout_index(&self, layout: &ArcLayout) -> Option<usize> {
        self.layout_index_map.get(layout).copied()
    }

    pub fn get_binary_layout_index(
        &self,
        from: &ArcLayout,
        to: &ArcLayout,
        permutation: Vec<usize>,
    ) -> Option<usize> {
        self.binary_layout_index_map
            .get(&(from.clone(), to.clone(), permutation))
            .copied()
    }

    pub fn get_translation_index(&self, from: &ArcLayout, to: &ArcLayout) -> Option<usize> {
        self.translate_index_map
            .get(&(from.clone(), to.clone()))
            .copied()
    }

    pub fn data(&self) -> &[f64] {
        self.data.as_ref()
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        self.data.as_mut_slice()
    }

    pub fn constants(&self) -> &[f64] {
        self.constants.as_ref()
    }

    pub fn constants_mut(&mut self) -> &mut [f64] {
        self.constants.as_mut_slice()
    }

    pub fn indices(&self) -> &[i32] {
        self.indices.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use approx::assert_relative_eq;

    use super::DataLayout;
    use crate::{discretise::DiscreteModel, parser::parse_ds_string};

    macro_rules! constant_layout_test {
        ($($name:ident: $text:literal expect $tensor_name:literal = $expected_value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let full_text = format!("
                    {}
                    u_i {{
                        y = 1,
                    }}
                    F_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", $text);

                let model = parse_ds_string(full_text.as_str()).unwrap();
                let discrete_model = match DiscreteModel::build("$name", &model) {
                    Ok(model) => model,
                    Err(e) => panic!("{}", e.as_error_message(full_text.as_str())),
                };
                let layout = DataLayout::new(&discrete_model).unwrap();
                let expected: Vec<f64> = $expected_value;
                assert_relative_eq!(
                    layout.get_tensor_constants($tensor_name).unwrap(),
                    expected.as_slice(),
                    epsilon = 1e-12
                );
            }
        )*
        }
    }

    macro_rules! constant_layout_new_panic_test {
        ($($name:ident: $text:literal,)*) => {
        $(
            #[test]
            #[should_panic]
            fn $name() {
                let full_text = format!("
                    {}
                    u_i {{
                        y = 1,
                    }}
                    F_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", $text);

                let model = parse_ds_string(full_text.as_str()).unwrap();
                let discrete_model = match DiscreteModel::build("$name", &model) {
                    Ok(model) => model,
                    Err(e) => panic!("{}", e.as_error_message(full_text.as_str())),
                };
                let _layout = DataLayout::new(&discrete_model).unwrap();
            }
        )*
        }
    }

    macro_rules! constant_layout_eval_panic_test {
        ($($name:ident: $text:literal select $getter:ident,)*) => {
        $(
            #[test]
            #[should_panic]
            fn $name() {
                let full_text = format!("
                    {}
                    u_i {{
                        y = 1,
                    }}
                    F_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", $text);

                let model = parse_ds_string(full_text.as_str()).unwrap();
                let discrete_model = match DiscreteModel::build("$name", &model) {
                    Ok(model) => model,
                    Err(e) => panic!("{}", e.as_error_message(full_text.as_str())),
                };
                let mut layout = DataLayout::new(&discrete_model).unwrap();
                let tensor = &discrete_model.$getter()[0];
                let blk = &tensor.elmts()[0];
                layout.evaluate_constant_block(tensor, blk);
            }
        )*
        }
    }

    constant_layout_test! {
        constant_dense_and_derived: "r_i { 2, 3 } k_i { 2 * r_i }" expect "k" = vec![4.0, 6.0],
        constant_sparse: "A_ij { (0, 1): 2, (1, 0): 3 }" expect "A" = vec![2.0, 3.0],
        constant_diagonal: "I_ij { (0..3, 0..3): 2 }" expect "I" = vec![2.0, 2.0, 2.0],
        constant_sparse_dense_add: "a_i { (0): 1, (2): 2 } b_i { (0:3): 3 } r_i { a_i + b_i }" expect "r" = vec![4.0, 3.0, 5.0],
        constant_sparse_dense_mul: "a_i { (0): 1, (2): 2 } b_i { (0:3): 3 } r_i { a_i * b_i }" expect "r" = vec![3.0, 6.0],
        constant_permuted_sparse_add: "A_ij { (0, 0): 1, (1, 1): 2 } B_ij { (0, 1): 3, (1, 1): 4 } R_ij { A_ij + B_ji }" expect "R" = vec![1.0, 3.0, 6.0],
        constant_dense_contraction: "a_ij { (0:2, 0:3): 2 } r_i { a_ij }" expect "r" = vec![6.0, 6.0],
        constant_sparse_contraction: "a_ijk { (0, 0, 0): 1, (1, 2, 3): 2 } r_ij { a_ijk }" expect "r" = vec![1.0, 2.0],
        constant_diagonal_contraction: "a_ijk { (0..3, 0..3, 0..3): 2 } r_ij { a_ijk }" expect "r" = vec![2.0, 2.0, 2.0],
        constant_broadcast_sparse_to_sparse: "A_i { (1): 2 } B_ij { (0:2, 0:2): A_i }" expect "B" = vec![2.0],
        constant_functions: "r_i { max(2, 3), pow(4, 0.5), arcsinh(1), heaviside(-0.1), sigmoid(0) }" expect "r" = vec![3.0, 2.0, 1.0_f64.asinh(), 0.0, 0.5],
    }

    constant_layout_new_panic_test! {
        constant_unknown_function_panics: "bad { definitely_not_a_function(1) }",
    }

    constant_layout_eval_panic_test! {
        constant_time_reference_panics: "bad { t }" select time_dep_defns,
        constant_input_reference_panics: "in { p = 1 } bad { p }" select input_dep_defns,
        constant_state_reference_panics: "bad_i { u_i }" select state_dep_defns,
    }

    fn write_temp_tns(name: &str, contents: &str) -> String {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("diffsl_data_layout_{name}_{unique}.tns"));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn constant_sparse_import() {
        let path = write_temp_tns(
            "constant_sparse_import",
            "
            2 3 5.0
            1 1 2.0
            ",
        );
        let full_text = format!(
            "
            C_ij {{ (0:3, 0:3): read('{}') }}
            u_i {{
                y = 1,
            }}
            F_i {{
                y,
            }}
            out_i {{
                y,
            }}
            ",
            path
        );

        let model = parse_ds_string(full_text.as_str()).unwrap();
        let discrete_model = match DiscreteModel::build("constant_sparse_import", &model) {
            Ok(model) => model,
            Err(e) => panic!("{}", e.as_error_message(full_text.as_str())),
        };
        let layout = DataLayout::new(&discrete_model).unwrap();
        assert_relative_eq!(
            layout.get_tensor_constants("C").unwrap(),
            &[2.0, 5.0][..],
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_interp1d_sorted_x() {
        let full_text = "
            xs_i { 0.0, 10.0, 20.0 }
            ys_i { 1.0, 5.0, 15.0 }
            u_i { z = 1 }
            F_i { z }
            r { interp1d(xs_i, ys_i, t) }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = match DiscreteModel::build("$name", &model) {
            Ok(model) => model,
            Err(e) => panic!("{}", e.as_error_message(full_text)),
        };
        DataLayout::new(&discrete_model).unwrap();
    }

    #[test]
    fn test_interp1d_unsorted_x_error() {
        let full_text = "
            xs_i { 10.0, 0.0, 20.0 }
            ys_i { 1.0, 5.0, 15.0 }
            u_i { z = 1 }
            F_i { z }
            r { interp1d(xs_i, ys_i, t) }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = match DiscreteModel::build("$name", &model) {
            Ok(model) => model,
            Err(e) => panic!("{}", e.as_error_message(full_text)),
        };
        let err = DataLayout::new(&discrete_model).unwrap_err();
        assert!(
            err.has_error_contains("strictly increasing"),
            "expected sortedness error, got: {}",
            err
        );
    }
}
