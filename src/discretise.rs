use core::panic;
use std::array;
use std::cell::RefCell;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::slice::SliceIndex;
use anyhow::{Result, anyhow};


use itertools::chain;
use ndarray::Array1;

use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Call;
use crate::ast::IndexedName;
use crate::builder::ModelInfo;
use crate::builder::Variable;
use crate::error::ValidationError;
use crate::error::ValidationErrors;

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct TensorBlock<'s> {
    start: Index,
    shape: Shape,
    expr: Ast<'s>,
}

impl<'s> TensorBlock<'s> {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn start(&self) -> &Index {
        &self.start
    }

    pub fn expr(&self) -> &Ast<'s> {
        &self.expr
    }
}

impl<'s> fmt::Display for TensorBlock<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct Tensor<'s> {
    name: &'s str,
    shape: Shape,
    elmts: Vec<TensorBlock<'s>>,
}

impl<'s> Tensor<'s> {
    pub fn new(name: &'s str) -> Self { 
        Self { 
            name, 
            shape: Shape::zeros(0), 
            elmts: Vec::new() 
        } 
    }
     pub fn push(&mut self, block: TensorBlock<'s>) {
        self.elmts.push(block);
     }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl<'s> fmt::Display for Tensor<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let elmts_str: Vec<String> = self.elmts.iter().map(|e| e.to_string()).collect();
        write!(f, "{} {{\n  {}\n}}", self.name, elmts_str.join("\n  "))
    }
}

#[derive(Debug)]
// the p[i] in F(t, p, u, u_dot) = G(t, p, u)
pub struct Input<'s> {
    name: &'s str,
    shape: Shape,
    domain: (f64, f64),
}

impl<'s> Input<'s> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl<'s> fmt::Display for Input<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let rank = self.shape().len();
        if rank > 1 {
            write!(f, "{}^{}", self.name, rank)
        } else {
            write!(f, "{}", self.name)
        }.and_then(|_|
            write!(f, " -> [{}, {}]", self.domain.0, self.domain.1)
        )
    }
}

#[derive(Debug)]
// the p[i] in F(t, p, u, u_dot) = G(t, p, u)
pub struct State<'s> {
    name: &'s str,
    shape: Shape,
    init: Option<Ast<'s>>,
}

impl<'s> State<'s> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl<'s> fmt::Display for State<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.init {
            Some(eq) => write!(f, "{} = {}", self.name, eq),
            None => write!(f, "{}", self.name),
        }
    }
}

type Shape = Array1<usize>;
type Index = Array1<i64>;

struct EnvVar {
    shape: Shape,
    is_time_dependent: bool,
    is_state_dependent: bool,
}

impl EnvVar {
    fn is_time_dependent(&self) -> bool {
        self.is_time_dependent
    }

    fn is_state_dependent(&self) -> bool {
        self.is_state_dependent
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

struct Env<'s> {
    errs: ValidationErrors,
    vars: HashMap<&'s str, EnvVar>,
}

pub fn broadcast_shapes(shapes: Vec<Shape>) -> Option<Shape> {
    if shapes.is_empty() {
        return None
    }
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap();
    let mut shape = Shape::zeros(max_rank);
    for i in max_rank-1..0 {
        let (mdim, compatible) = shapes
            .iter()
            .map(|s| s.get(i)
            .unwrap_or(&1))
            .fold((1, true), |(mdim, result), dim| {
                let new_mdim = max(mdim, *dim);
                (new_mdim, *dim == 1 || *dim == new_mdim)
            });
        if !compatible {
            return None
        }
        shape[i] = mdim;
    }
    Some(shape)
}

impl<'s> Env<'s> {
    pub fn new() -> Self {
        Env { errs: ValidationErrors::new(), vars: HashMap::new() }
    }
    pub fn is_tensor_time_dependent(&self, tensor: &Tensor) -> bool {
        tensor.elmts.iter().any(|block| {
            block.expr.get_dependents().iter().any(|&dep| {
                dep == "t" || self.vars[dep].is_time_dependent()
            })
        })
    }
    pub fn is_tensor_state_dependent(&self, tensor: &Tensor) -> bool {
        tensor.elmts.iter().any(|block| {
            block.expr.get_dependents().iter().any(|&dep| {
                dep == "u" || self.vars[dep].is_state_dependent()
            })
        })
    }

    pub fn push_var(&mut self, var: &Tensor) {
        self.vars.insert(var.name, EnvVar {
            shape: var.shape.clone(),
            is_time_dependent: self.is_tensor_time_dependent(var),
            is_state_dependent: self.is_tensor_state_dependent(var),
        });
    }

    fn get(&self, name: &str) -> Option<&EnvVar> {
        self.vars.get(name)
    }
    fn get_shape_binary_op(&self, left: &Ast, right: &Ast, indices: Vec<char>) -> Option<Shape> {
        let left_shape= self.get_shape(left, indices)?;
        let right_shape= self.get_shape(right, indices)?;
        match broadcast_shapes(vec!(left_shape, right_shape)) {
            Some(shape) => Some(shape),
            None => {
                self.errs.push(
                    ValidationError::new(
                        format!("cannot broadcast operands together. lhs {} and rhs {}", left_shape, right_shape),
                        left.span
                    )
                );
                None
            }
        }
        
    }
    fn get_shape_name(&mut self, name: &str, ast: &Ast, rhs_indices: Vec<char>, lhs_indices: Vec<char>) -> Option<Shape> {
        let var = self.get(name);
        if var.is_none() {
            self.errs.push(
                ValidationError::new(
                    format!("cannot find variable {}", name),
                    ast.span
                )
            );
            return None
        }
        let var = var.unwrap();
        let shape = var.shape();
        if rhs_indices.len() != shape.len() {
            self.errs.push(
                ValidationError::new(
                    format!("cannot index variable {} with {} indices. Expected {} indices", name, rhs_indices.len(), shape.len()),
                    ast.span
                )
            );
            return None
        }
        let mut new_shape = Shape::zeros(shape.len());
        for (rhs_index, c) in rhs_indices.iter().enumerate() {
            if let Some(lhs_index) = lhs_indices.iter().position(|&x| x == *c) {
                new_shape[lhs_index] = shape[rhs_index];
            } else {
                self.errs.push(
                    ValidationError::new(
                        format!("cannot find index {} in LHS indices {:?}", c, lhs_indices),
                        ast.span
                    )
                );
                return None
            }
        }
        Some(new_shape)
    }


    fn get_shape_sum(&mut self, call: &Call, ast: &Ast, indices: Vec<char>) -> Option<Shape> {
        if call.args.len() != 2 {
            self.errs.push(
                ValidationError::new(
                    format!("sum must have 2 arguments. found {}", call.args.len()),
                    ast.span
                )
            );
            return None
        }
        if call.args[0].kind.as_name().is_none() {
            self.errs.push(
                ValidationError::new(
                    format!("sum must have a variable as the first argument. found {}", call.args[0]),
                    ast.span
                )
            );
            return None
        }
        let name = call.args[0].kind.as_name().unwrap();
        if name.len() != 1 {
            self.errs.push(
                ValidationError::new(
                    format!("sum must have a single character variable as the first argument. found {}", name),
                    ast.span
                )
            );
            return None
        }
        let index = name.chars().next().unwrap();
        indices.push(index);
        self.get_shape(call.args[1].as_ref(), indices)
    }

    fn get_shape_call(&mut self, call: &Call<'s>, ast: &Ast, indices: Vec<char>) -> Option<Shape> {
        let shapes = call.args.iter().map(|c| self.get_shape(c, indices)).collect::<Option<Vec<Shape>>>()?; 
        match broadcast_shapes(shapes) {
            Some(shape) => Some(shape),
            None => {
                let shape_strs: Vec<String> = shapes.iter().map(|s| s.to_string()).collect();
                self.errs.push(
                    ValidationError::new(
                        format!("cannot broadcast operands together. shapes {:?}", shape_strs),
                        ast.span
                    )
                );
                None
            }
        }
    }

    pub fn get_shape(&mut self, ast: &Ast, indices: Vec<char>) -> Option<Shape> {
        match ast.kind {
            AstKind::Parameter(p) => self.get_shape(&p.domain, indices),
            AstKind::Assignment(a) => self.get_shape(a.expr.as_ref(), indices),
            AstKind::Binop(binop) => self.get_shape_binary_op(binop.left.as_ref(), binop.right.as_ref(), indices),
            AstKind::Monop(monop) => self.get_shape(monop.child.as_ref(), indices),
            AstKind::Call(call) => match call.fn_name {
                "sum" => self.get_shape_sum(&call, ast, indices),
                _ => self.get_shape_call(&call, ast, indices),
            }
            AstKind::CallArg(arg) => self.get_shape(arg.expression.as_ref(), indices),
            AstKind::Index(i) => self.get_shape_binary_op(i.left.as_ref(), i.right.as_ref(), indices),
            AstKind::Slice(s) => self.get_shape_binary_op(s.lower.as_ref(), s.upper.as_ref(), indices),
            AstKind::Number(n) => Some(Shape::zeros(0)),
            AstKind::Integer(i) => Some(Shape::zeros(0)),
            AstKind::Range(r) => Some(Shape::zeros(0)),
            AstKind::IndexedName(name) => self.get_shape_name(name.name, ast, name.indices, indices),
            AstKind::Name(name) => self.get_shape_name(name, ast, vec!(), indices),
            _ => panic!("unrecognised ast node {:#?}", ast.kind)
        }
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'s> {
    pub name: &'s str,
    pub lhs: Tensor<'s>,
    pub rhs: Tensor<'s>,
    pub out: Tensor<'s>,
    pub time_indep_defns: Vec<Tensor<'s>>,
    pub time_dep_defns: Vec<Tensor<'s>>,
    pub state_dep_defns: Vec<Tensor<'s>>,
    pub inputs: Vec<Input<'s>>,
    pub states: Vec<State<'s>>,
}

impl<'s, 'a> fmt::Display for DiscreteModel<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut inputs_str: Vec<String> = Vec::new();
        for input in self.inputs.iter() {
            inputs_str.push(format!("{}", input));
        }
        let mut states_str: Vec<String> = Vec::new();
        for state in self.states.iter() {
            states_str.push(format!("{}", state));
        }
        write!(f, "in {{\n  {}\n}}\n", inputs_str.join("\n  ")
        ).and_then(|_|
            self.time_indep_defns.iter().fold(Ok(()), |result, array| {
                result.and_then(|_| writeln!(f, "{}", array))
            })
        ).and_then(|_|
            write!(f, "u {{\n  {}\n}}\n", states_str.join("\n  "))
        ).and_then(|_|
            self.time_dep_defns.iter().fold(Ok(()), |result, array| {
                result.and_then(|_| writeln!(f, "{}", array))
            })
        ).and_then(|_|
            write!(f, "{}\n", self.lhs)
        ).and_then(|_|
            write!(f, "{}\n", self.rhs)
        ).and_then(|_|
            self.state_dep_defns.iter().fold(Ok(()), |result, array| {
                result.and_then(|_| writeln!(f, "{}", array))
            })
        ).and_then(|_|
            write!(f, "{}\n", self.out)
        )
    }
}



impl<'s> DiscreteModel<'s> {
    pub fn new(name: &'s str) -> Self { 
        Self {
            name,
            lhs: Tensor::new("F"),
            rhs: Tensor::new("G"),
            out: Tensor::new("out"),
            time_indep_defns: Vec::new(),
            time_dep_defns: Vec::new(),
            state_dep_defns: Vec::new(),
            inputs: Vec::new(),
            states: Vec::new(),
        }
    }


    fn build_states(tensor: &ast::Tensor, env: &mut Env) -> Vec<State<'s>> {
        let ret = Vec::new();
        assert_eq!(tensor.name, "u");
        for a in tensor.elmts {
            if let Some(elmt_shape) = env.get_shape(a.as_ref(), tensor.indices) {
                match a.kind {
                    AstKind::Assignment(ass) => {
                        let name = ass.name;
                        let shape = elmt_shape;
                        if shape.len() > 1 {
                            env.errs.push(
                                ValidationError::new(
                                    format!("state {} has shape {}, expected scalar or 1D array", name, shape),
                                    a.span
                                )
                            );
                        }
                        let init = Some(*ass.expr.clone());
                        ret.push(State{ name, shape, init })
                    }
                    _ => {
                        env.errs.push(
                            ValidationError::new(
                                format!("expected assignment in state definition"),
                                a.span
                            )
                        );
                    }
                }
            }
        }
        ret
    }
    
    fn build_inputs(tensor: &ast::Tensor, env: &mut Env) -> Vec<Input<'s>> {
        let ret = Vec::new();
        assert_eq!(tensor.name, "in");
        for a in tensor.elmts {
            if let Some(elmt_shape) = env.get_shape(a.as_ref(), tensor.indices) {
                match a.kind {
                    AstKind::Parameter(p) => {
                        let name = p.name;
                        let shape = elmt_shape;
                        if shape.len() > 1 {
                            env.errs.push(
                                ValidationError::new(
                                    format!("input shape must be a scalar or 1D vector"),
                                    a.span
                                )
                            );
                        }
                        let domain = match p.domain.kind {
                            AstKind::Range(r) => (r.lower, r.upper),
                            _ => {
                                env.errs.push(
                                    ValidationError::new(
                                        format!("expected range for parameter domain"),
                                        p.domain.span
                                    )
                                );
                                (0., 0.)
                            }
                        };
                        ret.push(Input{ name, shape, domain })
                    }
                    _ => {
                        env.errs.push(
                            ValidationError::new(
                                format!("expected parameter in input definition"),
                                a.span
                            )
                        );
                    }
                }
            }
        }
        ret
    }

    fn build_array(array: &ast::Tensor, env: &mut Env) -> Option<Tensor<'s>> {
        let rank = array.indices.len();
        let mut ret = Tensor::new(array.name);
        let mut start = Index::zeros(rank);
        if rank == 0 && array.elmts.len() > 1 {
            env.errs.push(
                ValidationError::new(
                    format!("cannot have more than one element in a scalar"),
                    array.elmts[1].span
                )
            );
        } else
        if rank > 2 && array.elmts.len() > 1 {
            env.errs.push(
                ValidationError::new(
                    format!("cannot have more than one element in a tensor with rank > 2"),
                    array.elmts[1].span
                )
            );
        }
        for a in array.elmts {
            if let Some(elmt_shape) = env.get_shape(a.as_ref(), array.indices) {
                ret.push(TensorBlock{ expr: *a.clone(), start: start, shape: elmt_shape });
                start = start + elmt_shape.mapv(|x| i64::try_from(x).unwrap());
            }
        }
        env.push_var(&ret);
        Some(ret)
    }

    pub fn build(name: &'s str, ast: &'s Vec<Box<Ast<'s>>>) -> Result<Self, ValidationErrors> {
        let mut env = Env::new();
        let ret = Self::new(name);
        let read_state= false;
        let read_F = false;
        let read_G = false;
        let read_out = false;
        for (i, tensor_ast) in ast.iter().enumerate() {
            match tensor_ast.kind.as_array() {
                None => env.errs.push(ValidationError::new("not an array".to_string(), tensor_ast.span)),
                Some(tensor) => {
                    let span = tensor_ast.span;
                    // first array must be in
                    if i == 0 &&  tensor.name != "in" {
                        env.errs.push(ValidationError::new("first array must be 'in'".to_string(), span));
                    }
                    match tensor.name {
                        "in" => {
                            Self::build_inputs(tensor, &mut env);
                        }
                        "u" => {
                            read_state = true;
                            Self::build_states(tensor, &mut env);
                        },
                        "F" => {
                            read_F = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.lhs.elmts.extend(built.elmts);
                            }
                        },
                        "G" => {
                            read_G = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.rhs.elmts.extend(built.elmts);
                            }
                        },
                        "out" => {
                            read_out = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                if built.shape.len() > 1 {
                                    env.errs.push(
                                        ValidationError::new(
                                            format!("output shape must be a scalar or 1D vector"),
                                            tensor_ast.span
                                        )
                                    );
                                }
                                ret.out.elmts.extend(built.elmts);
                            }
                        },
                        name => {
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                let env_entry = env.get(built.name).unwrap();
                                let dependent_on_state = env_entry.is_state_dependent();
                                let dependent_on_time = env_entry.is_time_dependent();
                                if !dependent_on_time {
                                    ret.time_indep_defns.push(built);
                                } else if dependent_on_time && !dependent_on_state {
                                    ret.time_dep_defns.push(built);
                                } else {
                                    ret.state_dep_defns.push(built);
                                }
                            }
                        }
                    }
                },
            }
        }
        Err(env.errs)
    }
    pub fn len_state(&self) -> usize {
        self.states.iter().fold(0, |sum, i| sum + i.shape()[0])
    }
    pub fn len_inputs(&self) -> usize {
        self.inputs.iter().fold(0, |sum, i| sum + i.shape()[0])
    }
    pub fn len_output(&self) -> usize {
        self.out.shape()[0]
    }
    pub fn get_init_state(&self) -> (Tensor<'s>, Tensor<'s>) {
        let alg_init = Ast {
            kind: AstKind::Number(0.),
            span: None,
        };
        (
            Tensor {
                name: "u0",
                elmts: self.states.iter().map(
                    |s| TensorBlock{ bounds: s.bounds, expr: match &s.init { Some(eq) => eq.clone(), None => alg_init.clone(), } } 
                ).collect(),
            },
            Tensor {
                name: "dotu0",
                elmts: self.states.iter().map(
                    |s| TensorBlock{ bounds: s.bounds, expr: alg_init.clone() } 
                ).collect(),
            }
        )
    }
    fn state_to_elmt(state_cell: &Rc<RefCell<Variable<'s>>>) -> (TensorBlock<'s>, TensorBlock<'s>) {
        let state = state_cell.borrow();
        let ast_eqn = if let Some(eqn) = &state.equation {
            eqn.clone()
        } else {
            panic!("state var should have an equation")
        };
        let (f_astkind, g_astkind) = match ast_eqn.kind {
            AstKind::RateEquation(eqn) => (
                AstKind::new_dot(Ast{ kind: AstKind::new_name(state.name), span: ast_eqn.span }),
                eqn.rhs.kind,
            ),
            AstKind::Equation(eqn) => (
                AstKind::new_num(0.0),
                AstKind::new_binop('-', *eqn.rhs, *eqn.lhs),
            ),
            _ => panic!("equation for state var should be rate eqn or standard eqn"),
        };
        (
            TensorBlock{ expr: Ast { kind: f_astkind, span: ast_eqn.span }, bounds: (0, u32::try_from(state.dim).unwrap()) },
            TensorBlock{ expr: Ast { kind: g_astkind, span: ast_eqn.span }, bounds: (0, u32::try_from(state.dim).unwrap()) },
        )
    }
    fn state_to_u0(state_cell: &Rc<RefCell<Variable<'s>>>) -> State<'s> {
        let state = state_cell.borrow();
        let init = if state.has_initial_condition() {
            Some(state.init_conditions[0].equation.clone())
        } else {
            None
        };
        State { name: state.name, bounds: (0, u32::try_from(state.dim).expect("cannot convert usize -> u32")), init }
    }
    fn idfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let defn = defn_cell.borrow();
        assert!(!defn.is_dependent_on_state());
        Tensor {
            name: defn.name,
            elmts: vec![TensorBlock {expr: defn.expression.as_ref().unwrap().clone(), bounds: (0, u32::try_from(defn.dim).unwrap()) }],
        }
    }
    fn odfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let defn = defn_cell.borrow();
        assert!(defn.is_dependent_on_state());
        Tensor {
            name: defn.name,
            elmts: vec![TensorBlock {expr: defn.expression.as_ref().unwrap().clone(), bounds: (0, u32::try_from(defn.dim).unwrap())}],
        }
    }
    
    fn state_to_input(input_cell: &Rc<RefCell<Variable<'s>>>) -> Input<'s> {
        let input = input_cell.borrow();
        assert!(input.is_independent());
        assert!(!input.is_time_dependent());
        Input {
            name: input.name,
            bounds: (0, u32::try_from(input.dim).unwrap()),
            domain: input.bounds,
        }
    }
    fn output_to_elmt(output_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let output = output_cell.borrow();
        TensorBlock {
            expr: Ast {
                kind: AstKind::new_name(output.name),
                span: if output.is_definition() { 
                    output.expression.as_ref().unwrap().span 
                } else if output.has_equation() { 
                    output.equation.as_ref().unwrap().span 
                } else { None } 
            },
            bounds: (0, u32::try_from(output.dim).unwrap())
        }
    }
    pub fn from(model: &ModelInfo<'s>) -> DiscreteModel<'s> {
        let (time_varying_unknowns, const_unknowns): (Vec<Rc<RefCell<Variable>>>, Vec<Rc<RefCell<Variable>>>)  = model.unknowns
            .iter()
            .cloned()
            .partition(|var| var.borrow().is_time_dependent());

        let states: Vec<Rc<RefCell<Variable>>> = time_varying_unknowns.iter().filter(|v| v.borrow().is_state()).cloned().collect();

        let (out_defns, in_defns): (Vec<Rc<RefCell<Variable>>>, Vec<Rc<RefCell<Variable>>>)  = model.definitions
            .iter()
            .cloned()
            .partition(|v| v.borrow().is_dependent_on_state());
        
        let mut out_array_elmts: Vec<TensorBlock> = chain(time_varying_unknowns.iter(), model.definitions.iter())
            .map(DiscreteModel::output_to_elmt).collect();
        let mut curr_index = 0;
        for elmt in out_array_elmts.iter_mut() {
            elmt.bounds.0 += curr_index;
            elmt.bounds.1 += curr_index;
            curr_index = elmt.bounds.1;
        }
        let out_array = Tensor {
            name: "out",
            elmts: out_array_elmts,
        };
        
        let mut f_elmts: Vec<TensorBlock> = Vec::new();
        let mut g_elmts: Vec<TensorBlock> = Vec::new();
        let mut curr_index = 0;
        let mut init_states: Vec<State> = Vec::new();
        for state in states.iter() {
            let mut elmt = DiscreteModel::state_to_elmt(state);
            elmt.0.bounds.0 += curr_index;
            elmt.0.bounds.1 += curr_index;
            elmt.1.bounds.0 += curr_index;
            elmt.1.bounds.1 += curr_index;
            let mut init_state = DiscreteModel::state_to_u0(state);
            init_state.bounds.0 += curr_index;
            init_state.bounds.1 += curr_index;
            curr_index = elmt.1.bounds.1;
            f_elmts.push(elmt.0);
            g_elmts.push(elmt.1);
            init_states.push(init_state);
        }
        
        let mut curr_index = 0;
        let mut inputs: Vec<Input> = Vec::new();
        for input in const_unknowns.iter() {
            let mut inp = DiscreteModel::state_to_input(input);
            inp.bounds.0 += curr_index;
            inp.bounds.1 += curr_index;
            curr_index = inp.bounds.1;
            inputs.push(inp);
        }
        
        let in_defns = in_defns.iter().map(DiscreteModel::idfn_to_array).collect();
        let out_defns = out_defns.iter().map(DiscreteModel::odfn_to_array).collect();
        let lhs = Tensor {
            name: "F",
            elmts: f_elmts,
        };
        let rhs = Tensor {
            name: "G",
            elmts: g_elmts,
        };
        let name = model.name;
        DiscreteModel {
            name,
            lhs, rhs,
            inputs,
            states: init_states,
            out: out_array,
            in_defns, out_defns
        }
    }
}




#[cfg(test)]
mod tests {
use crate::{ms_parser::parse_string, discretise::DiscreteModel, builder::ModelInfo};

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
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.in_defns.len(), 1);
        assert_eq!(discrete.in_defns[0].name, "inputVoltage");
        assert_eq!(discrete.out_defns.len(), 1);
        assert_eq!(discrete.out_defns[0].name, "doubleI");
        assert_eq!(discrete.lhs.name, "F");
        assert_eq!(discrete.rhs.name, "G");
        assert_eq!(discrete.states.len(), 1);
        assert_eq!(discrete.out.elmts.len(), 3);
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
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.out.elmts[0].expr.to_string(), "y");
        assert_eq!(discrete.out.elmts[1].expr.to_string(), "t");
        assert_eq!(discrete.out.elmts[2].expr.to_string(), "z");
        println!("{}", discrete);
    }
}
 