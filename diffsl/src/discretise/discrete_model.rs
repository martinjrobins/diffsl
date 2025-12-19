use anyhow::Result;
use log::info;
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
use super::Tensor;
use super::TensorBlock;
use super::ValidationError;
use super::ValidationErrors;

#[derive(Debug)]
// M(t, u_dot) = F(t, u)
pub struct DiscreteModel<'s> {
    name: &'s str,
    lhs: Option<Tensor<'s>>,
    rhs: Tensor<'s>,
    out: Option<Tensor<'s>>,
    constant_defns: Vec<Tensor<'s>>,
    input_dep_defns: Vec<Tensor<'s>>,
    time_dep_defns: Vec<Tensor<'s>>,
    state_dep_defns: Vec<Tensor<'s>>,
    dstate_dep_defns: Vec<Tensor<'s>>,
    inputs: Vec<Tensor<'s>>,
    state: Tensor<'s>,
    state_dot: Option<Tensor<'s>>,
    is_algebraic: Vec<bool>,
    stop: Option<Tensor<'s>>,
}

impl fmt::Display for DiscreteModel<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.inputs.is_empty() {
            write!(f, "in = [")?;
            for input in &self.inputs {
                write!(f, "{},", input.name())?;
            }
            writeln!(f, "]")?;
            for input in &self.inputs {
                writeln!(f, "{input}")?;
            }
        }
        for defn in &self.constant_defns {
            writeln!(f, "{defn}")?;
        }
        for defn in &self.input_dep_defns {
            writeln!(f, "{defn}")?;
        }
        for defn in &self.time_dep_defns {
            writeln!(f, "{defn}")?;
        }
        writeln!(f, "{}", self.state)?;
        if let Some(state_dot) = &self.state_dot {
            writeln!(f, "{state_dot}")?;
        }
        for defn in &self.state_dep_defns {
            writeln!(f, "{defn}")?;
        }
        if let Some(lhs) = &self.lhs {
            writeln!(f, "{lhs}")?;
        }
        writeln!(f, "{}", self.rhs)?;
        if let Some(stop) = &self.stop {
            writeln!(f, "{stop}")?;
        }
        if let Some(out) = &self.out {
            writeln!(f, "{out}")?;
        }
        Ok(())
    }
}

type VecVariable<'s> = Vec<Rc<RefCell<Variable<'s>>>>;

impl<'s> DiscreteModel<'s> {
    pub fn new(name: &'s str) -> Self {
        Self {
            name,
            lhs: None,
            rhs: Tensor::new_empty("F"),
            out: None,
            constant_defns: Vec::new(),
            input_dep_defns: Vec::new(),
            time_dep_defns: Vec::new(),
            state_dep_defns: Vec::new(),
            dstate_dep_defns: Vec::new(),
            inputs: Vec::new(),
            state: Tensor::new_empty("u"),
            state_dot: None,
            is_algebraic: Vec::new(),
            stop: None,
        }
    }

    fn build_array(
        array: &ast::Tensor<'s>,
        env: &mut Env,
        force_dense: bool,
    ) -> Option<Tensor<'s>> {
        let rank = array.indices().len();
        let reserved_names = [
            "u0",
            "t",
            "data",
            "root",
            "thread_id",
            "thread_dim",
            "rr",
            "states",
            "inputs",
            "outputs",
            "hass_mass",
        ];
        if reserved_names.contains(&array.name()) {
            let span = env.current_span().to_owned();
            env.errs_mut().push(ValidationError::new(
                format!("{} is a reserved name", array.name()),
                span,
            ));
            return None;
        }
        let mut elmts = Vec::new();
        let mut start = Index::zeros(rank);
        let nerrs = env.errs().len();
        if rank == 0 && array.elmts().len() > 1 {
            env.errs_mut().push(ValidationError::new(
                "cannot have more than one element in a scalar".to_string(),
                array.elmts()[1].span,
            ));
        }
        for a in array.elmts() {
            match &a.kind {
                AstKind::TensorElmt(te) => {
                    if let Some((expr_layout, elmt_layout)) =
                        env.get_layout_tensor_elmt(te, array.indices(), force_dense)
                    {
                        if rank == 0 && elmt_layout.rank() == 1 && elmt_layout.shape()[0] > 1 {
                            env.errs_mut().push(ValidationError::new(
                                format!("cannot assign an expression with rank > 1 to a scalar, rhs has shape {}", elmt_layout.shape()),
                                a.span,
                            ));
                        }
                        let (name, expr) = if let AstKind::Assignment(a) = &te.expr.kind {
                            (Some(String::from(a.name)), a.expr.clone())
                        } else {
                            (None, te.expr.clone())
                        };

                        // if the tensor indices indicates a start, use this, otherwise increment by the shape
                        if let Some(elmt_indices) = te.indices.as_ref() {
                            let given_indices_ast = &elmt_indices.kind.as_vector().unwrap().data;
                            let given_indices: Vec<&Indice> = given_indices_ast
                                .iter()
                                .map(|i| i.kind.as_indice().unwrap())
                                .collect();
                            start = Index::from_vec(
                                given_indices
                                    .into_iter()
                                    .map(|i| i.first.kind.as_integer().unwrap())
                                    .collect::<Vec<i64>>(),
                            )
                        }
                        let zero_axis_shape = if elmt_layout.rank() == 0 {
                            1
                        } else {
                            i64::try_from(elmt_layout.shape()[0]).unwrap()
                        };

                        if reserved_names
                            .contains(&name.as_ref().unwrap_or(&"".to_string()).as_str())
                        {
                            let span = env.current_span().to_owned();
                            env.errs_mut().push(ValidationError::new(
                                format!("{} is a reserved name", name.as_ref().unwrap()),
                                span,
                            ));
                        }

                        // make sure arc layouts are unique
                        // TODO: if we always use arc layout for the recursion, we can reuse existing ones
                        // much more efficiently rather than creating new ones all the time
                        let elmt_layout = env.new_layout_ptr(elmt_layout);
                        let expr_layout = if &expr_layout == elmt_layout.as_ref() {
                            // if layouts match, we can use the elmt layout ptr
                            elmt_layout.clone()
                        } else {
                            env.new_layout_ptr(expr_layout)
                        };

                        elmts.push(TensorBlock::new(
                            name,
                            start.clone(),
                            array.indices().to_vec(),
                            elmt_layout,
                            expr_layout,
                            *expr,
                        ));

                        // increment start index
                        if !start.is_empty() {
                            start[0] += zero_axis_shape;
                        }
                    }
                }
                _ => unreachable!("unexpected expression in tensor definition"),
            }
        }
        // create tensor
        if elmts.is_empty() {
            let span = env.current_span().to_owned();
            env.errs_mut().push(ValidationError::new(
                format!("tensor {} has no elements", array.name()),
                span,
            ));
            None
        } else {
            // todo: if we always use arc layout for the recursion, we can reuse existing ones
            match Layout::concatenate(&elmts, rank) {
                Ok(layout) => {
                    let layout = env.new_layout_ptr(layout);
                    let tensor = Tensor::new(array.name(), elmts, layout, array.indices().to_vec());
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
                    info!("Built tensor: {}", tensor);
                    Some(tensor)
                }
                Err(e) => {
                    let span = env.current_span().to_owned();
                    env.errs_mut()
                        .push(ValidationError::new(format!("{e}"), span));
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
        let mut env = Env::new(model.inputs.as_slice());
        let mut ret = Self::new(name);
        let mut read_state = false;
        let mut span_f = None;
        let mut span_m = None;
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
                            if let Some(built) = Self::build_array(tensor, &mut env, true) {
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
                            if let Some(built) = Self::build_array(tensor, &mut env, true) {
                                ret.state_dot = Some(built);
                            }
                            if ret.state.rank() > 1 {
                                env.errs_mut().push(ValidationError::new(
                                    "dudt must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                        }
                        "F" => {
                            if let Some(built) = Self::build_array(tensor, &mut env, true) {
                                span_f = Some(span);
                                ret.rhs = built;
                            }
                            // check that F is not dstatedt dependent and only depends on u
                            if let Some(f) = env.get("F") {
                                if f.is_dstatedt_dependent() {
                                    env.errs_mut().push(ValidationError::new(
                                        "F must not be dependent on dudt".to_string(),
                                        span,
                                    ));
                                }
                            }
                        }
                        "M" => {
                            if let Some(built) = Self::build_array(tensor, &mut env, true) {
                                span_m = Some(span);
                                ret.lhs = Some(built);
                            }
                            // check that M is not state dependent and only depends on dudt
                            if let Some(m) = env.get("M") {
                                if m.is_state_dependent() {
                                    env.errs_mut().push(ValidationError::new(
                                        "M must not be dependent on u".to_string(),
                                        span,
                                    ));
                                }
                            }
                        }
                        "stop" => {
                            if let Some(built) = Self::build_array(tensor, &mut env, true) {
                                ret.stop = Some(built);
                            }
                            // check that stop is not dependent on dudt
                            if let Some(stop) = env.get("stop") {
                                if stop.is_dstatedt_dependent() {
                                    env.errs_mut().push(ValidationError::new(
                                        "stop must not be dependent on dudt".to_string(),
                                        tensor_ast.span,
                                    ));
                                }
                            }
                        }
                        "out" => {
                            if let Some(built) = Self::build_array(tensor, &mut env, true) {
                                if built.rank() > 1 {
                                    env.errs_mut().push(ValidationError::new(
                                        "output shape must be a scalar or 1D vector".to_string(),
                                        tensor_ast.span,
                                    ));
                                }
                                ret.out = Some(built);
                            }
                            // check that out is not dependent on dudt
                            if let Some(out) = env.get("out") {
                                if out.is_dstatedt_dependent() {
                                    env.errs_mut().push(ValidationError::new(
                                        "out must not be dependent on dudt".to_string(),
                                        tensor_ast.span,
                                    ));
                                }
                            }
                        }
                        name => {
                            if let Some(built) = Self::build_array(tensor, &mut env, false) {
                                let is_input = model.inputs.contains(&name);
                                if let Some(env_entry) = env.get(built.name()) {
                                    let dependent_on_state = env_entry.is_state_dependent();
                                    let dependent_on_time = env_entry.is_time_dependent();
                                    let dependent_on_dudt = env_entry.is_dstatedt_dependent();
                                    let dependent_on_input = env_entry.is_input_dependent();
                                    if is_input {
                                        // inputs must be constants
                                        if dependent_on_time || dependent_on_state {
                                            env.errs_mut().push(ValidationError::new(
                                                format!("input {} must be constant", built.name()),
                                                tensor_ast.span,
                                            ));
                                        }
                                        ret.inputs.push(built);
                                    } else if !dependent_on_time && !dependent_on_input {
                                        ret.constant_defns.push(built);
                                    } else if !dependent_on_time {
                                        ret.input_dep_defns.push(built);
                                    } else if !dependent_on_state && !dependent_on_dudt {
                                        ret.time_dep_defns.push(built);
                                    } else if dependent_on_state {
                                        ret.state_dep_defns.push(built);
                                    } else if dependent_on_dudt {
                                        ret.dstate_dep_defns.push(built);
                                    } else {
                                        panic!("all the cases should be covered")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // reorder inputs to match the order defined in "in = [ ... ]"
        ret.inputs.sort_by_key(|t| {
            model
                .inputs
                .iter()
                .position(|&name| name == t.name())
                .unwrap()
        });

        // set is_algebraic for every state based on equations
        if ret.state_dot.is_some() && ret.lhs.is_some() {
            let state_dot = ret.state_dot.as_ref().unwrap();
            let lhs = ret.lhs.as_ref().unwrap();
            for i in 0..std::cmp::min(
                state_dot.elmts().len(),
                std::cmp::min(lhs.elmts().len(), ret.rhs.elmts().len()),
            ) {
                let sp = &state_dot.elmts()[i];
                let feq = lhs.elmts()[i].expr();
                let geq = ret.rhs.elmts()[i].expr();
                let geq_deps = geq.get_dependents();
                ret.is_algebraic.push(true);
                if let Some(sp_name) = sp.name() {
                    if Some(sp_name) == feq.kind.as_name().map(|n| n.name)
                        && !geq_deps.contains(sp_name)
                    {
                        ret.is_algebraic[i] = false;
                    }
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
                    format!("input {name} is not defined"),
                    span_all,
                ));
            }
        }

        // check that we've read all the required arrays
        if !read_state {
            env.errs_mut().push(ValidationError::new(
                "missing 'u' array".to_string(),
                span_all,
            ));
        }
        if span_f.is_none() {
            env.errs_mut().push(ValidationError::new(
                "missing 'F' array".to_string(),
                span_all,
            ));
        }
        if let Some(span) = span_f {
            Self::check_match(&ret.rhs, &ret.state, span, &mut env);
        }
        if let Some(span) = span_m {
            Self::check_match(ret.lhs.as_ref().unwrap(), &ret.state, span, &mut env);
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
                AstKind::new_time_derivative(state.name, vec![]),
                eqn.rhs.kind,
            ),
            AstKind::Equation(eqn) => (
                AstKind::new_num(0.0),
                AstKind::new_binop('-', *eqn.rhs, *eqn.lhs),
            ),
            _ => panic!("equation for state var should be rate eqn or standard eqn"),
        };
        (
            TensorBlock::new_dense_vector(
                None,
                0,
                state.dim,
                Ast {
                    kind: f_astkind,
                    span: ast_eqn.span,
                },
            ),
            TensorBlock::new_dense_vector(
                None,
                0,
                state.dim,
                Ast {
                    kind: g_astkind,
                    span: ast_eqn.span,
                },
            ),
        )
    }
    fn state_to_u0(state_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let state = state_cell.as_ref().borrow();
        let init = if state.has_initial_condition() {
            state.init_conditions[0].equation.clone()
        } else {
            Ast {
                kind: AstKind::new_num(0.0),
                span: None,
            }
        };
        TensorBlock::new_dense_vector(Some(state.name.to_owned()), 0, state.dim, init)
    }
    fn state_to_dudt0(state_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let state = state_cell.as_ref().borrow();
        let init = Ast {
            kind: AstKind::new_num(0.0),
            span: None,
        };
        let named_gradient_str = AstKind::new_time_derivative(state.name, vec![])
            .as_named_gradient()
            .unwrap()
            .to_string();
        TensorBlock::new_dense_vector(Some(named_gradient_str), 0, state.dim, init)
    }
    fn dfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let defn = defn_cell.as_ref().borrow();
        let tsr_blk = TensorBlock::new_dense_vector(
            None,
            0,
            defn.dim,
            defn.expression.as_ref().unwrap().clone(),
        );
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
            Ast {
                kind: AstKind::new_num(0.0),
                span: None,
            }
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
        let (time_varying_unknowns, const_unknowns): (VecVariable, VecVariable) = model
            .unknowns
            .iter()
            .cloned()
            .partition(|var| var.as_ref().borrow().is_time_dependent());

        let states: Vec<Rc<RefCell<Variable>>> = time_varying_unknowns
            .iter()
            .filter(|v| v.as_ref().borrow().is_state())
            .cloned()
            .collect();

        let (state_dep_defns, state_indep_defns): (VecVariable, VecVariable) = model
            .definitions
            .iter()
            .cloned()
            .partition(|v| v.as_ref().borrow().is_dependent_on_state());

        let (time_dep_defns, const_defns): (VecVariable, VecVariable) = state_indep_defns
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
            curr_index += init_state.layout().shape()[0];
            init_dudts.push(init_dudt);
            init_states.push(init_state);
        }

        let state = Tensor::new_no_layout("u", init_states, vec!['i']);
        let state_dot = Tensor::new_no_layout("dudt", init_dudts, vec!['i']);

        // define F and G
        let mut curr_index = 0;
        let mut m_elmts: Vec<TensorBlock> = Vec::new();
        let mut f_elmts: Vec<TensorBlock> = Vec::new();
        let mut is_algebraic = Vec::new();
        for state in states.iter() {
            let mut elmt = DiscreteModel::state_to_elmt(state);
            elmt.0.start_mut()[0] = i64::try_from(curr_index).unwrap();
            elmt.1.start_mut()[0] = i64::try_from(curr_index).unwrap();
            curr_index += elmt.0.layout().shape()[0];
            m_elmts.push(elmt.0);
            f_elmts.push(elmt.1);
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
        let constant_defns = const_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let lhs = Tensor::new_no_layout("M", m_elmts, vec!['i']);
        let rhs = Tensor::new_no_layout("F", f_elmts, vec!['i']);
        let name = model.name;
        let stop = None;
        let dstate_dep_defns = Vec::new();
        DiscreteModel {
            name,
            lhs: Some(lhs),
            rhs,
            inputs,
            state,
            state_dot: Some(state_dot),
            out: Some(out_array),
            constant_defns,
            input_dep_defns: Vec::new(), // todo: need to implement
            time_dep_defns,
            state_dep_defns,
            dstate_dep_defns,
            is_algebraic,
            stop,
        }
    }

    pub fn inputs(&self) -> &[Tensor<'_>] {
        self.inputs.as_ref()
    }

    pub fn constant_defns(&self) -> &[Tensor<'_>] {
        self.constant_defns.as_ref()
    }

    pub fn input_dep_defns(&self) -> &[Tensor<'_>] {
        self.input_dep_defns.as_ref()
    }

    pub fn time_dep_defns(&self) -> &[Tensor<'_>] {
        self.time_dep_defns.as_ref()
    }
    pub fn state_dep_defns(&self) -> &[Tensor<'_>] {
        self.state_dep_defns.as_ref()
    }

    pub fn dstate_dep_defns(&self) -> &[Tensor<'_>] {
        self.dstate_dep_defns.as_ref()
    }

    pub fn state(&self) -> &Tensor<'s> {
        &self.state
    }

    pub fn state_dot(&self) -> Option<&Tensor<'s>> {
        self.state_dot.as_ref()
    }

    pub fn out(&self) -> Option<&Tensor<'s>> {
        self.out.as_ref()
    }

    pub fn lhs(&self) -> Option<&Tensor<'s>> {
        self.lhs.as_ref()
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

    pub fn stop(&self) -> Option<&Tensor<'_>> {
        self.stop.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        continuous::ModelInfo,
        discretise::DiscreteModel,
        execution::Translation,
        parser::{parse_ds_string, parse_ms_string},
    };

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
        assert_eq!(discrete.input_dep_defns().len(), 0);
        assert_eq!(discrete.constant_defns().len(), 0);
        assert_eq!(discrete.time_dep_defns.len(), 1);
        assert_eq!(discrete.time_dep_defns[0].name(), "inputVoltage");
        assert_eq!(discrete.state_dep_defns.len(), 1);
        assert_eq!(discrete.state_dep_defns[0].name(), "doubleI");
        assert_eq!(discrete.lhs().unwrap().name(), "M");
        assert_eq!(discrete.rhs.name(), "F");
        assert_eq!(discrete.state.shape()[0], 1);
        assert_eq!(discrete.state.elmts().len(), 1);
        assert_eq!(discrete.out().unwrap().elmts().len(), 3);
        println!("{discrete}");
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
        assert_eq!(discrete.out().unwrap().elmts()[0].expr().to_string(), "y");
        assert_eq!(discrete.out().unwrap().elmts()[1].expr().to_string(), "t");
        assert_eq!(discrete.out().unwrap().elmts()[2].expr().to_string(), "z");
        println!("{discrete}");
    }

    #[test]
    fn tensor_classification() {
        let text = "
            in = [r, k, ]
            r { 1, }
            k { 1, }
            z { 2 * r }
            g { 2 * t }
            u_i {
                y = 1,
                z = 0,
            }
            u2_i {
                2 * y,
                2 * z,
            }
            dudt_i {
                dydt = 0,
                dzdt = 0,
            }
            dudt2_i {
                2 * dydt,
                0,
            }
            M_i {
                dydt,
                0,
            }
            F_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                y,
                t,
                z,
            }
        ";
        let model = parse_ds_string(text).unwrap();
        let model = DiscreteModel::build("$name", &model).unwrap();
        assert_eq!(
            model.inputs().iter().map(|t| t.name()).collect::<Vec<_>>(),
            ["r", "k"]
        );
        assert_eq!(
            model
                .constant_defns()
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>(),
            Vec::<&str>::new()
        );
        assert_eq!(
            model
                .input_dep_defns()
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>(),
            ["z"]
        );
        assert_eq!(
            model
                .time_dep_defns()
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>(),
            ["g"]
        );
        assert_eq!(
            model
                .state_dep_defns()
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>(),
            ["u2"]
        );
        assert_eq!(
            model
                .dstate_dep_defns()
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>(),
            ["dudt2"]
        );
        assert_eq!(
            model.inputs().iter().map(|t| t.name()).collect::<Vec<_>>(),
            ["r", "k"]
        );
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
            M_i {
                dydt,
                0,
            }
            F_i {
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
            M {
                dydt,
            }
            F {
                (r * y) * (1 - y),
            }
            out {
                y,
            }
        " [],
        logistic_no_m: "
            in = [r, ]
            r { 1, }    
            u {
                y = 1,
            }
            F {
                (r * y) * (1 - y),
            }
            out {
                y,
            }
        " [],
        logistic_no_m2: "
            in = [r, ]
            r { 1, }    
            u_i {
                x = 1,
                y = 1,
            }
            F_i {
                (r * x) * (1 - x),
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
            M_i {
                dydt,
            }
            F_i {
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
            M_i {
                dydt_i,
                0,
                0,
            }
            F_i {
                I_ij * rhs_i,
            }
            out_i {
                y_i,
                t,
                z_i,
            }
        " [],
        error_missing_specials: "" ["missing 'u' array", "missing 'F' array",],
        error_state_lhs_rhs_same: "
            u_i {
                y = 1,
            }
            F_i {
                y,
                1,
            }
        " ["F and u must have the same shape",],
        error_dep_on_dudt: "
            u_i {
                y = 1,
            }
            dudt_i {
                dydt = 0,
            }
            F_i {
                dydt,
            }
            stop_i {
                dydt,
            }
            out_i {
                dydt,
            }
        " ["F must not be dependent on dudt", "stop must not be dependent on dudt", "out must not be dependent on dudt",],
        error_m_dep_on_u: "
            u_i {
                y = 1,
            }
            y2_i {
                2 * y,
            }
            dudt_i {
                dydt = 0,
            }
            M_i {
                y2_i,
            }
            F_i {
                y,
            }
            out_i {
                y,
            }
        " ["M must not be dependent on u",],
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
                    F_i {{
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
                    F_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", tensor_text);
                let model = parse_ds_string(model_text.as_str()).unwrap();
                match DiscreteModel::build("$name", &model) {
                    Ok(model) => {
                        let tensor = model.constant_defns().iter().chain(model.time_dep_defns.iter()).find(|t| t.name() == $tensor_name).unwrap();
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
        error_different_shape: "a_i { 1, 2 } b_i { 1, 2, 3 } c_i { a_i + b_i }" errors ["cannot broadcast shapes: [2], [3]",],
        too_many_indices: "A_i { 1, 2 } B_i { (0:2): A_ij }" errors ["too many permutation indices",],
        bcast_expr_to_elmt: "A_i { 1, 2 } B_i { (0:2): A_i, (2:3): A_i }" errors ["cannot broadcast expression shape [2] to tensor element shape [1]",],
        error_index1: "A_ij { (0:3, 0:3): 1.0 } B_i { A_ij[1:3] }" errors ["can only index dense 1D variables",],
        error_index2: "A_i { (2): 1.0 } B_i { A_i[1:3] }" errors ["can only index dense 1D variables",],
        error_index3: "A_i { 0.0, 1.0, 2.0 } B_i { (0:1): A_i[1:3] }" errors ["cannot broadcast expression shape [2] to tensor element shape [1]",],
        error_index4: "A { 1.0 } B { A[0] }" errors ["can only index dense 1D variables",],
        error_contract_1d_to_scalar: "A_i { 1.0, 2.0 } B { A_i }" errors ["contraction only supported from 2D to 1D tensors. Got 1D to 0D",],
        error_broadcast_vect_matrix: "A_ij { (0:3, 0:2): 1.0 } b_i { (0:2): 1.0 } c_ij { A_ij + b_i }" errors ["cannot broadcast shapes: [3, 2], [2]",],
        error_divide_by_zero: "a_i { (0): 1, (2): 2 } b_i { (2): 1 } c_i { a_i / b_i }" errors ["divide-by-zero",],
        error_divide_by_zero2: "a_i { (0:3): 1 } b_i { (2): 1 } c_i { a_i / b_i }" errors ["divide-by-zero",],
        slice_sparse_vec: "A_i { (0): 1, (2): 3 } B_i { A_i[0:1] }" errors ["can only index dense 1D variables",],
    );

    tensor_tests!(
        sparse_dense_concat: "a_i { (0): 1, (2): 3 } b_i { 4, 5 } r_i { a_i, b_i }" expect "r" = "r_i (5s) { (0)(3s): a_i (3s) , (3)(2): b_i (2) }",
        exp_sparse_vec: "a_i { (0): 1, (2): 3 } r_i { exp(a_i) }" expect "r" = "r_i (3) { (0)(3): exp(a_i) (3)}",
        max_sparse_scalar: "a_i { (0): 1, (2): 3 } r_i { max(a_i, 2) }" expect "r" = "r_i (3) { (0)(3): max(a_i, 2) (3) }",
        sparse_mat_vec_mul: "A_ij { (1, 1): 2 } b_j { (1): 3 } r_i { A_ij * b_j }" expect "r" = "r_i (2s) { (0)(2s): A_ij * b_j (2s,2s) }",
        sparse_broadcast_to_sparse: "A_i { (1): 2 } B_ij { (0:2, 0:2): A_i }" expect "B" = "B_ij (2s,2) { (0,0)(2s,2): A_i (2s) }",
        sparse_contract_to_sparse: "A_ij { (1, 1): 2 } B_i { A_ij }" expect "B" = "B_i (2s) { (0)(2s): A_ij (2s,2s) }",
        diag_sparse_add: "a_ij { (0..2, 0..2): 2 } b_ij { (1, 1): 3 } c_ij { a_ij + b_ij }" expect "c" = "c_ij (2i,2i) { (0,0)(2i,2i): a_ij + b_ij (2i,2i) }",
        diag_dense_mul: "a_ij { (0..2, 0..2): 2 } b_ij { (0:2, 0:2): 3 } c_ij { a_ij * b_ij }" expect "c" = "c_ij (2i,2i) { (0,0)(2i,2i): a_ij * b_ij (2i,2i) }",
        diag_dense_add: "a_ij { (0..2, 0..2): 2 } b_ij { (0:2, 0:2): 3 } c_ij { a_ij + b_ij }" expect "c" = "c_ij (2,2) { (0,0)(2,2): a_ij + b_ij (2,2) }",
        sparse_dense_mat_add: "a_ij { (2, 2): 2 } b_ij { (0:3, 0:3): 3 } c_ij { a_ij + b_ij }" expect "c" = "c_ij (3,3) { (0,0)(3,3): a_ij + b_ij (3,3) }",
        sparse_dense_mat_mul: "a_ij { (2, 2): 2 } b_ij { (0:3, 0:3): 3 } c_ij { a_ij * b_ij }" expect "c" = "c_ij (3s,3s) { (0,0)(3s,3s): a_ij * b_ij (3s,3s) }",
        sparse_dense_mat_mul2: "a_ij { (0, 0): 2, (1, 1): 1 } b_ij { (0:2, 0:2): 3 } c_ij { a_ij * b_ij }" expect "c" = "c_ij (2i,2i) { (0,0)(2i,2i): a_ij * b_ij (2i,2i) }",
        sparse_dense_vec_add: "a_i { (2): 2 } b_i { (0:3): 3 } c_i { a_i + b_i }" expect "c" = "c_i (3) { (0)(3): a_i + b_i (3) }",
        sparse_dense_vec_add2: "a_i { (2): 2 } b_i { (0:3): 3 } c_i { b_i + a_i }" expect "c" = "c_i (3) { (0)(3): b_i + a_i (3) }",
        sparse_sparse_vec_add:  "a_i { (2): 2 } b_i { (0): 3, (2): 4 } c_i { a_i + b_i }" expect "c" = "c_i (3s) { (0)(3s): a_i + b_i (3s) }",
        sparse_sparse_vec_add2:  "a_i { (2): 2 } b_i { (0): 3, (2): 4 } c_i { b_i + a_i }" expect "c" = "c_i (3s) { (0)(3s): b_i + a_i (3s) }",
        sparse_sparse_vec_add3: "a_i { (1): 1, (2): 2 } b_i { (0): 3, (2): 4 } c_i { a_i + b_i }" expect "c" = "c_i (3) { (0)(3): a_i + b_i (3) }",
        sparse_dense_vec_mul: "a_i { (2): 2 } b_i { (0:3): 3 } c_i { a_i * b_i }" expect "c" = "c_i (3s) { (0)(3s): a_i * b_i (3s) }",
        sparse_dense_vec_mul2: "a_i { (2): 2 } b_i { (0:3): 3 } c_i { b_i * a_i }" expect "c" = "c_i (3s) { (0)(3s): b_i * a_i (3s) }",
        two_dim_sparse_add: "A_ij { (0, 0): 1, (1, 0): 1, (1, 1): 1 } B_ij { (1, 1): 1 } C_ij { A_ij + B_ij }" expect "C" = "C_ij (2s,2s) { (0, 0)(2s,2s): A_ij + B_ij (2s,2s) }",
        mat_mul_sparse_vec: "A_ij { (0, 0): 1, (1, 0): 2, (1, 1): 3 } x_i { (1): 1 } b_i { A_ij * x_j }" expect "b" = "b_i (2s) { (0)(2s): A_ij * x_j (2s, 2s) }",
        add_sparse_vecs: "a_i { (2): 3 } b_i { (1): 2, (2): 4 } c_i { a_i + b_i }" expect "c" = "c_i (3s) { (0)(3s): a_i + b_i (3s) }",
        add_sparse_vecs_to_dense: "a_i { (0): 1, (2): 3 } b_i { (1): 2, (2): 4 } c_i { a_i + b_i }" expect "c" = "c_i (3) { (0)(3): a_i + b_i (3) }",
        row_vec: "a_ij { (0, 0): 1, (0, 1): 2 } b_i { (0:3): 1 } c_i { a_ij * b_j[0:2] }" expect "c" = "c_i (1) { (0)(1): a_ij * b_j[0:2] (1, 2) }",
        col_vec: "a_ij { (0, 0): 1, (1, 0): 2 } b_i { (0:2): a_ij }" expect "b" = "b_i (2) { (0)(2): a_ij (2, 1) }",
        broadcast_vect_matrix: "A_ij { (0:3, 0:2): 1.0 } b_i { (0:2): 1.0 } c_ij { A_ij + b_j }" expect "c" = "c_ij (3,2) { (0,0)(3,2): A_ij + b_j (3,2) }",
        contract_2d_to_1d: "A_ij { (0:3, 0:3): 1.0 } B_i { A_ij }" expect "B" = "B_i (3) { (0)(3): A_ij (3, 3) }",
        index: "A_i { 0.0, 1.0, 2.0 } B { A_i[1] }" expect "B" = "B { (): A_i[1] }",
        index2: "A_i { 0.0, 1.0, 2.0 } B_i { A_i[1:3] }" expect "B" = "B_i(2) { (0)(2):A_i[1:3](2) }",
        index3: "A_ij { (0:2, 0:2): 1 } g_i { 0, 1, 2 } b_i { A_ij * g_j[0:2] }" expect "b" = "b_i (2) { (0)(2): A_ij * g_j[0:2] (2, 2) }",
        prefix_minus: "A { 1.0 / -2.0 }" expect "A" = "A { (): 1 / (-2) }",
        time: "A_i { t }" expect "A" = "A_i (1) { (0)(1):  t }",
        named_blk: "A_i { (0:3): y = 1, 2 }" expect "A" = "A_i (4) { (0)(3): y = 1, (3)(1): 2 }",
        dense_vect_implicit: "A_i { 1, 2, 3 }" expect "A" = "A_i (3) { (0)(1): 1, (1)(1): 2, (2)(1): 3 }",
        dense_vect_explicit: "A_i { (0:3): 1, (3:4): 2 }" expect "A" = "A_i (4) { (0)(3): 1, (3)(1): 2 }",
        dense_vect_mix: "A_i { (0:3): 1, 2 }" expect "A" = "A_i (4) { (0)(3): 1, (3)(1): 2 }",
        diag_matrix: "A_ij { (0, 0): 1, (1, 1): 4 }" expect "A" = "A_ij (2i,2i) { (0, 0)(1, 1): 1, (1, 1)(1, 1): 4 }",
        sparse_matrix: "A_ij { (0, 0): 1, (0, 1): 2, (1, 1): 4 }" expect "A" = "A_ij (2s,2s) { (0, 0)(1, 1): 1, (0, 1)(1, 1): 2, (1, 1)(1, 1): 4 }",
        sparse_row_matrix: "A_ij { (0, 1): 2, (0, 2): 4 }" expect "A" = "A_ij (1s,3s) { (0, 1)(1, 1): 2, (0, 2)(1, 1): 4 }",
        same_sparsity: "A_i { (0): 1, (1): 1, (3): 1 } B_i { (0): 2, (1): 3, (3): 4 } C_i { A_i + B_i, }" expect "C" = "C_i (4s) { (0)(4s): A_i + B_i (4s) }",
        diagonal: "A_ij { (0..2, 0..2): 1 } " expect "A" = "A_ij (2i,2i) { (0, 0)(2i, 2i): 1 }",
        concat_diags: "A_ij { (0..2, 0..2): 1 } B_ij { (0:2, 0:2): A_ij, (2, 2): 1 }" expect "B" = "B_ij (3i,3i) { (0, 0)(2i,2i): A_ij (2i, 2i), (2, 2)(1, 1): 1 }",
        sparse_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 0): 2, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" = "b_i (2) { (0)(2): A_ij * x_j (2s, 2s) }",
        diag_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" = "b_i (2) { (0)(2): A_ij * x_j (2i, 2i) }",
        dense_matrix_vect_multiply: "A_ij {  (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" = "b_i (2) { (0)(2): A_ij * x_j (2, 2) }",
        sparse_matrix_vect_multiply_zero_row: "A_ij { (0, 0): 1, (0, 1): 2 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" = "b_i (1) { (0)(1): A_ij * x_j (1, 2) }",
        mat_mul_sparse_vec_out: "A_ij { (1, 0): 2, (1, 1): 3 } x_i { (0:2): 1 } b_i { A_ij * x_j }" expect "b" = "b_i (2s) { (0)(2s): A_ij * x_j (2s, 2s) }",

    );

    #[test]
    fn test_stop() {
        let text_no_stop = "
        u_i {
            y = 1,
        }
        F_i {
            y * (1 - y),
        }
        out {
            y,
        }
        ";
        let text_stop = text_no_stop.to_owned() + "stop_i { y - 0.5 }";
        let model_ds_no_stop = parse_ds_string(text_no_stop).unwrap();
        let model_ds = parse_ds_string(text_stop.as_str()).unwrap();
        let model_no_stop = DiscreteModel::build("$name", &model_ds_no_stop).unwrap();
        let model = DiscreteModel::build("$name", &model_ds).unwrap();
        assert!(model_no_stop.stop().is_none());
        assert_eq!(
            model.stop().unwrap().elmts()[0].expr().to_string(),
            "y - 0.5"
        );
        assert_eq!(model.stop().unwrap().name(), "stop");
        assert_eq!(model.stop().unwrap().elmts().len(), 1);
    }

    #[test]
    fn test_no_out() {
        let text = "
        u_i {
            y = 1,
        }
        F_i {
            y * (1 - y),
        }
        ";
        let model = parse_ds_string(text).unwrap();
        let model = DiscreteModel::build("$name", &model).unwrap();
        assert!(model.out().is_none());
    }

    #[test]
    fn test_constants_and_input_dep() {
        let text = "
        in = [r]
        r { 1, }
        k { 1, }
        r2 { 2 * r }
        u_i {
            y = k,
        }
        F_i {
            r * y,
        }
        ";
        let model = parse_ds_string(text).unwrap();
        let model = DiscreteModel::build("$name", &model).unwrap();
        assert_eq!(
            model
                .constant_defns()
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>(),
            ["k"]
        );
        assert_eq!(
            model
                .input_dep_defns()
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>(),
            ["r2"]
        );
    }

    #[test]
    fn test_sparse_layout() {
        let text = "
        u_i {
            y = 1,
        }
        r_ij {
            (0..3, 0..3): 1,
            (1..3, 0..2): 3,
        }
        b_ij {
            (0, 0): 1,
            (1, 0): 3,
            (1, 1): 1,
            (2, 1): 3,
            (2, 2): 1,
        }
        F_i {
            y,
        }
        ";
        let model = parse_ds_string(text).unwrap();
        let model = DiscreteModel::build("$name", &model).unwrap();
        let r = model
            .constant_defns()
            .iter()
            .find(|t| t.name() == "r")
            .unwrap();
        let b = model
            .constant_defns()
            .iter()
            .find(|t| t.name() == "b")
            .unwrap();
        for tensor in [r, b] {
            let layout = tensor.layout();
            assert_eq!(layout.shape()[0], 3);
            assert_eq!(layout.shape()[1], 3);
            assert_eq!(
                layout.indices().map(|i| i.to_string()).collect::<Vec<_>>(),
                vec!["[0, 0]", "[1, 0]", "[1, 1]", "[2, 1]", "[2, 2]"]
            );
            assert_eq!(layout.to_data_layout(), vec![0, 0, 1, 0, 1, 1, 2, 1, 2, 2]);
        }
        let translation = Translation::new(
            r.elmts()[0].expr_layout(),
            r.elmts()[0].layout(),
            r.elmts()[0].start(),
            r.layout_ptr(),
        );
        assert_eq!(translation.to_data_layout(), vec![0, 2, 4]);
        let translation = Translation::new(
            r.elmts()[1].expr_layout(),
            r.elmts()[1].layout(),
            r.elmts()[1].start(),
            r.layout_ptr(),
        );
        assert_eq!(translation.to_data_layout(), vec![1, 3]);
    }
}
