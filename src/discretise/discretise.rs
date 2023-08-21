use anyhow::Result;
use core::panic;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::vec;

use itertools::chain;

use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Indice;
use crate::ast::StringSpan;
use crate::continuous::ModelInfo;
use crate::continuous::Variable;

use super::Env;
use super::Index;
use super::Layout;
use super::RcLayout;
use super::Tensor;
use super::TensorBlock;
use super::ValidationError;
use super::ValidationErrors;


#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'s> {
    name: &'s str,
    lhs: Tensor<'s>,
    rhs: Tensor<'s>,
    out: Tensor<'s>,
    time_indep_defns: Vec<Tensor<'s>>,
    time_dep_defns: Vec<Tensor<'s>>,
    state_dep_defns: Vec<Tensor<'s>>,
    inputs: Vec<Tensor<'s>>,
    state: Tensor<'s>,
    state_dot: Tensor<'s>,
    is_algebraic: Vec<bool>,
}

impl<'s, 'a> fmt::Display for DiscreteModel<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.inputs.len() > 0 {
            write!(f, "in = [")?;
            for input in &self.inputs {
                write!(f, "{},", input.name())?;
            }
            write!(f, "]\n")?;
            for input in &self.inputs {
                write!(f, "{}\n", input)?;
            }
        }
        for defn in &self.time_indep_defns {
            write!(f, "{}\n", defn)?;
        }
        for defn in &self.time_dep_defns {
            write!(f, "{}\n", defn)?;
        }
        write!(f, "{}\n", self.state)?;
        write!(f, "{}\n", self.state_dot)?;
        for defn in &self.state_dep_defns {
            write!(f, "{}\n", defn)?;
        }
        write!(f, "{}\n", self.lhs)?;
        write!(f, "{}\n", self.rhs)?;
        write!(f, "{}\n", self.out)
    }
}

impl<'s> DiscreteModel<'s> {
    pub fn new(name: &'s str) -> Self {
        Self {
            name,
            lhs: Tensor::new_empty("F"),
            rhs: Tensor::new_empty("G"),
            out: Tensor::new_empty("out"),
            time_indep_defns: Vec::new(),
            time_dep_defns: Vec::new(),
            state_dep_defns: Vec::new(),
            inputs: Vec::new(),
            state: Tensor::new_empty("u"),
            state_dot: Tensor::new_empty("u_dot"),
            is_algebraic: Vec::new(),
        }
    }

    // residual = F(t, u, u_dot) - G(t, u)
    // return a tensor equal to the residual
    pub fn residual(&self) -> Tensor<'s> {
        let mut residual = self.lhs.clone();
        residual.set_name("residual");
        let lhs = Ast {
            kind: AstKind::new_indexed_name("F", vec!['i']),
            span: None,
        };
        let rhs = Ast {
            kind: AstKind::new_indexed_name("G", vec!['i']),
            span: None,
        };
        let name = "residual";
        let indices = self.lhs.indices().to_vec();
        let layout = self.lhs.layout_ptr().clone();
        let elmts = vec![
            TensorBlock::new(
                None,
                Index::from_vec(vec![0]),
                indices.clone(),
                self.lhs.layout_ptr().clone(),
                self.lhs.layout_ptr().clone(),
                Ast {
                    kind: AstKind::new_binop('-', lhs, rhs),
                    span: None,
                },
            )
        ];
        Tensor::new(name, elmts, layout, indices)
    }

    

    fn build_array(array: &ast::Tensor<'s>, env: &mut Env) -> Option<Tensor<'s>> {
        let rank = array.indices().len();
        let mut elmts = Vec::new();
        let mut start = Index::zeros(rank);
        let nerrs = env.errs().len();
        if rank == 0 && array.elmts().len() > 1 {
            env.errs_mut().push(ValidationError::new(
                format!("cannot have more than one element in a scalar"),
                array.elmts()[1].span,
            ));
        }
        for a in array.elmts() {
            match &a.kind {
                AstKind::TensorElmt(te) => {
                    if let Some((expr_layout, elmt_layout)) = env.get_layout_tensor_elmt(&te, array.indices()) {
                        if rank == 0 && elmt_layout.rank() == 1 {
                            if elmt_layout.shape()[0] > 1 {
                                env.errs_mut().push(ValidationError::new(
                                    format!("cannot assign an expression with rank > 1 to a scalar, rhs has shape {}", elmt_layout.shape()),
                                    a.span,
                                ));
                            }
                        }
                        let (name, expr) = if let AstKind::Assignment(a) = &te.expr.kind {
                            (Some(String::from(a.name)), a.expr.clone())
                        } else {
                            (None, te.expr.clone())
                        };
                        
                        // if the tensor indices indicates a start, use this, otherwise increment by the shape
                        if let Some(elmt_indices) = te.indices.as_ref() {
                            let given_indices_ast = &elmt_indices.kind.as_vector().unwrap().data;
                            let given_indices: Vec<&Indice> = given_indices_ast.iter().map(|i| i.kind.as_indice().unwrap()).collect();
                            start = Index::from_vec(given_indices.into_iter().map(|i| i.first.kind.as_integer().unwrap()).collect::<Vec<i64>>())
                        }
                        let zero_axis_shape= if elmt_layout.rank() == 0 {
                            1
                        } else {
                            i64::try_from(elmt_layout.shape()[0]).unwrap()
                        };
                        
                        elmts.push(TensorBlock::new(name, start.clone(), array.indices().to_vec(), RcLayout::new(elmt_layout), RcLayout::new(expr_layout), *expr));

                        // increment start index
                        if start.len() > 0 {
                            start[0] += zero_axis_shape;
                        }
                    }
                },
                _ => unreachable!("unexpected expression in tensor definition"),
            }
        }
        // create tensor 
        if elmts.is_empty() {
            let span = env.current_span().to_owned();
            env.errs_mut().push(ValidationError::new(
                format!("tensor {} has no elements", array.name()),
                span 
            ));
            None
        } else {
            match Layout::concatenate(&elmts, rank) {
                Ok(layout) => {
                    let tensor = Tensor::new(array.name(), elmts, RcLayout::new(layout), array.indices().to_vec());
                    //check that the number of indices matches the rank
                    assert_eq!(tensor.rank(), tensor.indices().len());
                    if nerrs == env.errs().len() {
                        env.push_var(&tensor);
                        for block in tensor.elmts().iter() {
                            if let Some(_name) = block.name() {
                                env.push_var_blk(&tensor, block);
                            }
                        }
                    }
                    Some(tensor)
                },
                Err(e) => {
                    let span = env.current_span().to_owned();
                    env.errs_mut().push(ValidationError::new(
                        format!("{}", e),
                        span 
                    ));
                    None
                }
            }
        }
    }


    fn check_match(tensor1: &Tensor, tensor2: &Tensor, span: Option<StringSpan>, env: &mut Env) {
        // check shapes
        if tensor1.shape() != tensor2.shape() {
            env.errs_mut().push(ValidationError::new(
                format!(
                    "{} and {} must have the same shape, but {} has shape {} and {} has shape {}",
                    tensor1.name(),
                    tensor2.name(),
                    tensor1.name(),
                    tensor1.shape(),
                    tensor2.name(),
                    tensor2.shape()
                ),
                span,
            ));
        }
    }

    pub fn build(name: &'s str, model: &'s ast::DsModel) -> Result<Self, ValidationErrors> {
        let mut env = Env::new();
        let mut ret = Self::new(name);
        let mut read_state = false;
        let mut read_dot_state = false;
        let mut read_out = false;
        let mut span_f = None;
        let mut span_g = None;
        for tensor_ast in model.tensors.iter() {
            env.set_current_span(tensor_ast.span);
            match tensor_ast.kind.as_array() {
                None => env.errs_mut().push(ValidationError::new(
                    "not an array".to_string(),
                    tensor_ast.span,
                )),
                Some(tensor) => {
                    let span = tensor_ast.span;
                    // if env has a tensor with the same name, error
                    if env.get(tensor.name()).is_some() {
                        env.errs_mut().push(ValidationError::new(
                            format!("{} is already defined", tensor.name()),
                            span,
                        ));
                    }
                    match tensor.name() {
                        "u" => {
                            read_state = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.state = built;
                            }
                            if ret.state.rank() > 1 {
                                env.errs_mut().push(ValidationError::new(
                                    "u must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                        }
                        "dudt" => {
                            read_dot_state = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.state_dot = built;
                            }
                            if ret.state.rank() > 1 {
                                env.errs_mut().push(ValidationError::new(
                                    "dudt must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                        }
                        "F" => {
                            span_f = Some(span);
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.lhs = built;
                            }
                        }
                        "G" => {
                            span_g = Some(span);
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.rhs = built;
                            }
                        }
                        "out" => {
                            read_out = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                if built.rank() > 1 {
                                    env.errs_mut().push(ValidationError::new(
                                        format!("output shape must be a scalar or 1D vector"),
                                        tensor_ast.span,
                                    ));
                                }
                                ret.out = built;
                            }
                        }
                        _name => {
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                let is_input = model.inputs.iter().any(|name| *name == _name);
                                if let Some(env_entry) = env.get(built.name()) {
                                    let dependent_on_state = env_entry.is_state_dependent();
                                    let dependent_on_time = env_entry.is_time_dependent();
                                    if is_input {
                                        // inputs must be constants
                                        if dependent_on_time || dependent_on_state {
                                            env.errs_mut().push(ValidationError::new(
                                                format!("input {} must be constant", built.name()),
                                                tensor_ast.span,
                                            ));
                                        }
                                        ret.inputs.push(built);
                                    } else if !dependent_on_time {
                                        ret.time_indep_defns.push(built);
                                    } else if dependent_on_time && !dependent_on_state {
                                        ret.time_dep_defns.push(built);
                                    } else {
                                        ret.state_dep_defns.push(built);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        

        // set is_algebraic for every state based on equations
        for i in 0..std::cmp::min(ret.state_dot.elmts().len(), std::cmp::min(ret.lhs.elmts().len(), ret.rhs.elmts().len())) {
            let sp = &ret.state_dot.elmts()[i];
            let feq = ret.lhs.elmts()[i].expr();
            let geq = ret.rhs.elmts()[i].expr();
            let geq_deps = geq.get_dependents();
            ret.is_algebraic.push(true);
            if let Some(sp_name) = sp.name() {
                if Some(sp_name) == feq.kind.as_name() && !geq_deps.contains(sp_name) {
                    ret.is_algebraic[i] = false;
                }
            }
        }
        
        let span_all = if model.tensors.is_empty() {
            None
        } else {
            Some(StringSpan {
                pos_start: model.tensors.first().unwrap().span.unwrap().pos_start,
                pos_end: model.tensors.last().unwrap().span.unwrap().pos_start,
            })
        };
        // check that we found all input parameters
        for name in model.inputs.iter() {
            if env.get(name).is_none() {
                env.errs_mut().push(ValidationError::new(
                    format!("input {} is not defined", name),
                    span_all,
                ));
            }
        };

        // check that we've read all the required arrays
        if !read_state {
            env.errs_mut().push(ValidationError::new(
                "missing 'u' array".to_string(),
                span_all,
            ));
        }
        if !read_dot_state {
            env.errs_mut().push(ValidationError::new(
                "missing 'dudt' array".to_string(),
                span_all,
            ));
        }
        if span_f.is_none() {
            env.errs_mut().push(ValidationError::new(
                "missing 'F' array".to_string(),
                span_all,
            ));
        }
        if span_g.is_none() {
            env.errs_mut().push(ValidationError::new(
                "missing 'G' array".to_string(),
                span_all,
            ));
        }
        if !read_out {
            env.errs_mut().push(ValidationError::new(
                "missing 'out' array".to_string(),
                span_all,
            ));
        }
        if let Some(span) = span_f {
            Self::check_match(&ret.lhs, &ret.state, span, &mut env);
        }
        if let Some(span) = span_g {
            Self::check_match(&ret.rhs, &ret.state, span, &mut env);
        }

        if env.errs().is_empty() {
            Ok(ret)
        } else {
            Err(env.errs().to_owned())
        }
    }
    
    fn state_to_elmt(state_cell: &Rc<RefCell<Variable<'s>>>) -> (TensorBlock<'s>, TensorBlock<'s>) {
        let state = state_cell.as_ref().borrow();
        let ast_eqn = if let Some(eqn) = &state.equation {
            eqn.clone()
        } else {
            panic!("state var should have an equation")
        };
        let (f_astkind, g_astkind) = match ast_eqn.kind {
            AstKind::RateEquation(eqn) => (
                AstKind::new_time_derivative(state.name),
                eqn.rhs.kind,
            ),
            AstKind::Equation(eqn) => (
                AstKind::new_num(0.0),
                AstKind::new_binop('-', *eqn.rhs, *eqn.lhs),
            ),
            _ => panic!("equation for state var should be rate eqn or standard eqn"),
        };
        (
            TensorBlock::new_dense_vector(None, 0, state.dim, Ast { kind: f_astkind, span: ast_eqn.span }),
            TensorBlock::new_dense_vector(None, 0, state.dim, Ast { kind: g_astkind, span: ast_eqn.span }),
        )
    }
    fn state_to_u0(state_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let state = state_cell.as_ref().borrow();
        let init = if state.has_initial_condition() {
            state.init_conditions[0].equation.clone()
        } else {
            Ast { kind: AstKind::new_num(0.0), span: None }
        };
        TensorBlock::new_dense_vector(Some(state.name.to_owned()), 0, state.dim, init)
    }
    fn state_to_dudt0(state_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let state = state_cell.as_ref().borrow();
        let init = Ast { kind: AstKind::new_num(0.0), span: None };
        let named_gradient_str = AstKind::new_time_derivative(state.name).as_named_gradient().unwrap().to_string();
        TensorBlock::new_dense_vector(Some(named_gradient_str), 0, state.dim, init)
    }
    fn dfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let defn = defn_cell.as_ref().borrow();
        let tsr_blk = TensorBlock::new_dense_vector(None, 0, defn.dim, defn.expression.as_ref().unwrap().clone());
        let layout = tsr_blk.layout().clone();
        Tensor::new(defn.name, vec![tsr_blk], layout, vec!['i'])
    }

    fn state_to_input(input_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let input = input_cell.as_ref().borrow();
        assert!(input.is_independent());
        assert!(!input.is_time_dependent());
        let expr = if let Some(expr) = &input.expression {
            expr.clone()
        } else {
            Ast { kind: AstKind::new_num(0.0), span: None }
        };
        let elmt = TensorBlock::new_dense_vector(None, 0, input.dim, expr);
        let indices = vec!['i'];
        Tensor::new_no_layout(input.name, vec![elmt], indices)
    }
    fn output_to_elmt(output_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let output = output_cell.as_ref().borrow();
        let expr = Ast {
            kind: AstKind::new_name(output.name),
            span: if output.is_definition() {
                output.expression.as_ref().unwrap().span
            } else if output.has_equation() {
                output.equation.as_ref().unwrap().span
            } else {
                None
            },
        };
        TensorBlock::new_dense_vector(None, 0, output.dim, expr)
    }
    pub fn from(model: &ModelInfo<'s>) -> DiscreteModel<'s> {
        let (time_varying_unknowns, const_unknowns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = model
            .unknowns
            .iter()
            .cloned()
            .partition(|var| var.as_ref().borrow().is_time_dependent());

        let states: Vec<Rc<RefCell<Variable>>> = time_varying_unknowns
            .iter()
            .filter(|v| v.as_ref().borrow().is_state())
            .cloned()
            .collect();

        let (state_dep_defns, state_indep_defns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = model
            .definitions
            .iter()
            .cloned()
            .partition(|v| v.as_ref().borrow().is_dependent_on_state());

        let (time_dep_defns, const_defns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = state_indep_defns
            .iter()
            .cloned()
            .partition(|v| v.as_ref().borrow().is_time_dependent());

        let mut out_array_elmts: Vec<TensorBlock> =
            chain(time_varying_unknowns.iter(), model.definitions.iter())
                .map(DiscreteModel::output_to_elmt)
                .collect();

        // fix out start indices
        let mut curr_index = 0;
        for elmt in out_array_elmts.iter_mut() {
            elmt.start_mut()[0] = i64::try_from(curr_index).unwrap();
            curr_index += elmt.layout().shape()[0];
        }
        let out_array = Tensor::new_no_layout("out", out_array_elmts, vec!['i']);

        // define u and dtdt
        let mut curr_index = 0;
        let mut init_states: Vec<TensorBlock> = Vec::new();
        let mut init_dudts: Vec<TensorBlock> = Vec::new();
        for state in states.iter() {
            let mut init_state = DiscreteModel::state_to_u0(state);
            let mut init_dudt = DiscreteModel::state_to_dudt0(state);
            init_state.start_mut()[0] = i64::try_from(curr_index).unwrap();
            init_dudt.start_mut()[0] = i64::try_from(curr_index).unwrap();
            curr_index = curr_index + init_state.layout().shape()[0];
            init_dudts.push(init_dudt);
            init_states.push(init_state);
        }

        let state = Tensor::new_no_layout("u", init_states, vec!['i']);
        let state_dot = Tensor::new_no_layout("dudt", init_dudts, vec!['i']);

        // define F and G
        let mut curr_index = 0;
        let mut f_elmts: Vec<TensorBlock> = Vec::new();
        let mut g_elmts: Vec<TensorBlock> = Vec::new();
        let mut is_algebraic = Vec::new();
        for state in states.iter() {
            let mut elmt = DiscreteModel::state_to_elmt(state);
            elmt.0.start_mut()[0] = i64::try_from(curr_index).unwrap();
            elmt.1.start_mut()[0] = i64::try_from(curr_index).unwrap();
            curr_index = curr_index + elmt.0.layout().shape()[0];
            f_elmts.push(elmt.0);
            g_elmts.push(elmt.1);
            is_algebraic.push(state.as_ref().borrow().is_algebraic().unwrap());
        }

        let mut inputs: Vec<Tensor> = Vec::new();
        for input in const_unknowns.iter() {
            let inp = DiscreteModel::state_to_input(input);
            inputs.push(inp);
        }

        let state_dep_defns = state_dep_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let time_dep_defns = time_dep_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let time_indep_defns = const_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let lhs =  Tensor::new_no_layout("F", f_elmts, vec!['i']);
        let rhs = Tensor::new_no_layout("G", g_elmts, vec!['i']);
        let name = model.name;
        DiscreteModel {
            name,
            lhs,
            rhs,
            inputs,
            state,
            state_dot,
            out: out_array,
            time_indep_defns,
            time_dep_defns,
            state_dep_defns,
            is_algebraic,
        }
    }

    pub fn inputs(&self) -> &[Tensor] {
        self.inputs.as_ref()
    }

    pub fn time_indep_defns(&self) -> &[Tensor] {
        self.time_indep_defns.as_ref()
    }
    pub fn time_dep_defns(&self) -> &[Tensor] {
        self.time_dep_defns.as_ref()
    }
    pub fn state_dep_defns(&self) -> &[Tensor] {
        self.state_dep_defns.as_ref()
    }

    pub fn state(&self) -> &Tensor<'s> {
        &self.state
    }

    pub fn state_dot(&self) -> &Tensor<'s> {
        &self.state_dot
    }

    pub fn out(&self) -> &Tensor<'s> {
        &self.out
    }

    pub fn lhs(&self) -> &Tensor<'s> {
        &self.lhs
    }

    pub fn rhs(&self) -> &Tensor<'s> {
        &self.rhs
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn is_algebraic(&self) -> &[bool] {
        self.is_algebraic.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use crate::{parser::{parse_ms_string, parse_ds_string}, continuous::ModelInfo, discretise::DiscreteModel};


    #[test]
    fn test_circuit_model() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i(t)) {
            let inputVoltage = sin(t) 
            use resistor(v = inputVoltage)
            let doubleI = 2 * i
        }
        ";
        let models = parse_ms_string(text).unwrap();
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.time_indep_defns.len(), 0);
        assert_eq!(discrete.time_dep_defns.len(), 1);
        assert_eq!(discrete.time_dep_defns[0].name(), "inputVoltage");
        assert_eq!(discrete.state_dep_defns.len(), 1);
        assert_eq!(discrete.state_dep_defns[0].name(), "doubleI");
        assert_eq!(discrete.lhs.name(), "F");
        assert_eq!(discrete.rhs.name(), "G");
        assert_eq!(discrete.state.shape()[0], 1);
        assert_eq!(discrete.state.elmts().len(), 1);
        assert_eq!(discrete.out.elmts().len(), 3);
        println!("{}", discrete);
    }
    #[test]
    fn rate_equation() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), t, z(t) ) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
            z = 2 * y
        }
        ";
        let models = parse_ms_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.out.elmts()[0].expr().to_string(), "y");
        assert_eq!(discrete.out.elmts()[1].expr().to_string(), "t");
        assert_eq!(discrete.out.elmts()[2].expr().to_string(), "z");
        println!("{}", discrete);
    }

    macro_rules! count {
        () => (0usize);
        ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
    }

    macro_rules! full_model_tests {
        ($($name:ident: $text:literal [$($error:literal,)*],)*) => {
        $(
            #[test]
            fn $name() {
                let model_text = $text;
                let model = parse_ds_string(model_text).unwrap();
                match DiscreteModel::build("$name", &model) {
                    Ok(model) => {
                        if (count!($($error)*) != 0) {
                            panic!("Should have failed: {}", model)
                        }
                    }
                    Err(e) => {
                        if (count!($($error)*) == 0) {
                            panic!("Should have succeeded: {}", e.as_error_message(model_text))
                        } else {
                            $(
                                if !e.has_error_contains($error) {
                                    panic!("Expected error '{}' not found in '{}'", $error, e.as_error_message(model_text));
                                }
                            )*
                        }
                    }
                };
            }
        )*
        }
    }

    full_model_tests! (
        logistic: "
            in = [r, k, ]
            r { 1, }
            k { 1, }
            u_i {
                y = 1,
                z = 0,
            }
            dudt_i {
                dydt = 0,
                dzdt = 0,
            }
            F_i {
                dydt,
                0,
            }
            G_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                y,
                t,
                z,
            }
        " [],
        logistic_single_state: "
            in = [r, ]
            r { 1, }    
            u {
                y = 1,
            }
            dudt {
                dydt = 0,
            }
            F {
                dydt,
            }
            G {
                (r * y) * (1 - y),
            }
            out {
                y,
            }
        " [],
        scalar_state_as_vector: "
            in = [r, ]
            r { 1, }    
            u_i {
                y = 1,
            }
            dudt_i {
                dydt = 0,
            }
            F_i {
                dydt,
            }
            G_i {
                (r * y) * (1 - y),
            }
            out {
                y,
            }
        " [],
        logistic_matrix: "
            in = [r, k,]
            r { 1, }
            k { 1, }
            sm_ij {
                (0..2, 0..2): 1,
            }
            I_ij {
                (0:2, 0:2): sm_ij,
                (2, 2): 1,
                (3, 3): 1,
            }
            u_i {
                (0:2): y = 1,
                (2:4): z = 0,
            }
            dudt_i {
                (0:2): dydt = 0,
                (2:4): dzdt = 0,
            }
            rhs_i {
                (r * y_i) * (1 - (y_i / k)),
                (2 * y_i) - z_i,
            }
            F_i {
                dydt_i,
                0,
                0,
            }
            G_i {
                I_ij * rhs_i,
            }
            out_i {
                y_i,
                t,
                z_i,
            }
        " [],
        error_missing_specials: "" ["missing 'u' array", "missing 'dudt' array", "missing 'F' array", "missing 'G' array", "missing 'out' array",],
        error_state_lhs_rhs_same: "
            u_i {
                y = 1,
            }
            G_i {
                y,
                1,
            }
        " ["G and u must have the same shape",],
    );

    
    macro_rules! tensor_fail_tests {
        ($($name:ident: $text:literal errors [$($error:literal,)*],)*) => {
        $(
            #[test]
            fn $name() {
                let tensor_text = $text;
                let model_text = format!("
                    {}
                    u_i {{
                        y = 1,
                    }}
                    dudt_i {{
                        dydt = 0,
                    }}
                    F_i {{
                        dydt,
                    }}
                    G_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", tensor_text);
                let model = parse_ds_string(model_text.as_str()).unwrap();
                match DiscreteModel::build("$name", &model) {
                    Ok(model) => {
                        if (count!($($error)*) != 0) {
                            panic!("Should have failed: {}", model)
                        }
                    }
                    Err(e) => {
                        if (count!($($error)*) == 0) {
                            panic!("Should have succeeded: {}", e.as_error_message(model_text.as_str()))
                        } else {
                            $(
                                if !e.has_error_contains($error) {
                                    panic!("Expected error '{}' not found in '{}'", $error, e.as_error_message(model_text.as_str()));
                                }
                            )*
                        }
                    }
                };
            }
        )*
        }
    }

    macro_rules! tensor_tests {
        ($($name:ident: $text:literal expect $tensor_name:literal = $tensor_string:literal,)*) => {
        $(
            #[test]
            fn $name() {
                let tensor_text = $text;
                let model_text = format!("
                    {}
                    u_i {{
                        y = 1,
                    }}
                    dudt_i {{
                        dydt = 0,
                    }}
                    F_i {{
                        dydt,
                    }}
                    G_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", tensor_text);
                let model = parse_ds_string(model_text.as_str()).unwrap();
                match DiscreteModel::build("$name", &model) {
                    Ok(model) => {
                        let tensor = model.time_indep_defns.iter().find(|t| t.name() == $tensor_name).unwrap();
                        let tensor_string = format!("{}", tensor).chars().filter(|c| !c.is_whitespace()).collect::<String>();
                        let tensor_check_string = $tensor_string.chars().filter(|c| !c.is_whitespace()).collect::<String>();
                        assert_eq!(tensor_string, tensor_check_string);
                    }
                    Err(e) => {
                        panic!("Should have succeeded: {}", e.as_error_message(model_text.as_str()))
                    }
                };
            }
        )*
        }
    }

    tensor_fail_tests!(
        error_input_not_defined: "in = [bub]" errors ["input bub is not defined",],
        error_scalar: "r {1, 2}" errors ["cannot have more than one element in a scalar",],
        error_cannot_find: "r { k }" errors ["cannot find variable k",],
        error_different_sparsity: "A_ij { (0, 0): 1, (1, 0): 1, (1, 1): 1 } B_ij { (1, 1): 1 } C_ij { A_ij + B_ij }" errors ["cannot broadcast layouts with different sparsity",],
        error_different_shape: "a_i { 1, 2 } b_i { 1, 2, 3 } c_i { a_i + b_i }" errors ["cannot broadcast shapes: [2], [3]",],
        too_many_indices: "A_i { 1, 2 } B_i { (0:2): A_ij }" errors ["too many permutation indices",],
        bcast_expr_to_elmt: "A_i { 1, 2 } B_i { (0:2): A_i, (2:3): A_i }" errors ["cannot broadcast expression shape [2] to tensor element shape [1]",],
    );

    tensor_tests!(
        named_blk: "A_i { (0:3): y = 1, 2 }" expect "A" = "A_i (4) { (0)(3): y = 1, (3)(1): 2 }",
        dense_vect_implicit: "A_i { 1, 2, 3 }" expect "A" = "A_i (3) { (0)(1): 1, (1)(1): 2, (2)(1): 3 }",
        dense_vect_explicit: "A_i { (0:3): 1, (3:4): 2 }" expect "A" = "A_i (4) { (0)(3): 1, (3)(1): 2 }",
        dense_vect_mix: "A_i { (0:3): 1, 2 }" expect "A" = "A_i (4) { (0)(3): 1, (3)(1): 2 }",
        dense_matrix: "A_ij { (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 }" expect "A" = "A_ij (2,2) { (0, 0)(1, 1): 1, (0, 1)(1, 1): 2, (1, 0)(1, 1): 3, (1, 1)(1, 1): 4 }",
        diag_matrix: "A_ij { (0, 0): 1, (1, 1): 4 }" expect "A" = "A_ij (2i,2i) { (0, 0)(1, 1): 1, (1, 1)(1, 1): 4 }",
        sparse_matrix: "A_ij { (0, 0): 1, (0, 1): 2, (1, 1): 4 }" expect "A" = "A_ij (2s,2s) { (0, 0)(1, 1): 1, (0, 1)(1, 1): 2, (1, 1)(1, 1): 4 }",
        same_sparsity: "A_i { (0): 1, (1): 1, (3): 1 } B_i { (0): 2, (1): 3, (3): 4 } C_i { A_i + B_i, }" expect "C" = "C_i (4s) { (0)(4s): A_i + B_i (4s) }",
        diagonal: "A_ij { (0..2, 0..2): 1 } " expect "A" = "A_ij (2i,2i) { (0, 0)(2i, 2i): 1 }",
        concat_diags: "A_ij { (0..2, 0..2): 1 } B_ij { (0:2, 0:2): A_ij, (2, 2): 1 }" expect "B" = "B_ij (3i,3i) { (0, 0)(2i,2i): A_ij (2i, 2i), (2, 2)(1, 1): 1 }",
        sparse_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 0): 2, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" = "b_i (2) { (0)(2): A_ij * x_j (2s, 2s) }",
        diag_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" = "b_i (2) { (0)(2): A_ij * x_j (2i, 2i) }",
        dense_matrix_vect_multiply: "A_ij {  (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" = "b_i (2) { (0)(2): A_ij * x_j (2, 2) }",
    );
}
