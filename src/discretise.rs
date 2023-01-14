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

use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::builder::ModelInfo;
use crate::builder::Variable;
use crate::error::ValidationError;
use crate::error::ValidationErrors;

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct ArrayElmt<'s> {
    pub bounds: (u32, u32),
    pub expr: Ast<'s>,
}

impl<'s> ArrayElmt<'s> {
    pub fn get_shape(&self) -> u32 {
        self.bounds.1 - self.bounds.0
    }
}

impl<'s> fmt::Display for ArrayElmt<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct Array<'s> {
    pub name: &'s str,
    pub elmts: Vec<ArrayElmt<'s>>,
}

impl<'s> Array<'s> {
    pub fn new(name: &'s str) -> Self { Self { name, elmts: Vec::new() } }

    pub fn get_shape(&self) -> u32 {
        self.elmts.iter().fold(0, |sum, e| sum + e.get_shape())
    }
}

impl<'s> fmt::Display for Array<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let elmts_str: Vec<String> = self.elmts.iter().map(|e| e.to_string()).collect();
        write!(f, "{} {{\n  {}\n}}", self.name, elmts_str.join("\n  "))
    }
}

#[derive(Debug)]
// the p[i] in F(t, p, u, u_dot) = G(t, p, u)
pub struct Input<'s> {
    pub name: &'s str,
    pub bounds: (u32, u32),
    pub domain: (f64, f64),
}

impl<'s> fmt::Display for Input<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dim = self.get_shape();
        if dim > 1 {
            write!(f, "{}^{}", self.name, dim)
        } else {
            write!(f, "{}", self.name)
        }.and_then(|_|
            write!(f, " -> [{}, {}]", self.domain.0, self.domain.1)
        )
    }
}

impl<'s> Input<'s> {
    pub fn get_shape(&self) -> u32 {
        self.bounds.1 - self.bounds.0
    }
}

#[derive(Debug)]
// the p[i] in F(t, p, u, u_dot) = G(t, p, u)
pub struct State<'s> {
    pub name: &'s str,
    pub bounds: (u32, u32),
    init: Option<Ast<'s>>,
}

impl<'s> fmt::Display for State<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.init {
            Some(eq) => write!(f, "{} = {}", self.name, eq),
            None => write!(f, "{}", self.name),
        }
    }
}

impl<'s> State<'s> {
    pub fn get_shape(&self) -> u32 {
        self.bounds.1 - self.bounds.0
    }
    pub fn is_algebraic(&self) ->bool {
        self.init.is_none()
    }
}

#[derive(Debug, PartialEq)]
struct Shape {
    data: Vec<usize>,
}

impl Shape {
    pub fn rank(&self) -> usize {
        self.data.len()
    }
    pub fn get<I>(&self, index: I) -> Option<&<I as SliceIndex<[usize]>>::Output> 
    where 
        I: std::slice::SliceIndex<[usize]>
    {
        self.data.get(index)
    }

    fn new_with_rank(max_rank: usize) -> Self {
        Shape { data: vec![1; max_rank] }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = write!(f, "[");
        let n = self.data.len();
        for i in 0..n {
            if i == n-1 {
                result = write!(f, "{}]", self.data[i])
            } else {
                result = write!(f, "{}, ", self.data[i])
            }
        }
        result
    }
}


struct Env<'s> {
    errs: ValidationErrors,
    vars: HashMap<&'s str, Shape>,
}

pub fn broadcast_shapes(shapes: Vec<Shape>) -> Option<Shape> {
    if shapes.is_empty() {
        return None
    }
    let max_rank = shapes.iter().map(|s| s.rank()).max().unwrap();
    let mut shape = Shape::new_with_rank(max_rank);
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
    pub fn get_shape(&mut self, ast: &Ast) -> Option<Shape> {
        match ast.kind {
            AstKind::Binop(binop) => {
                let left_shape= self.get_shape(binop.left.as_ref())?;
                let right_shape= self.get_shape(binop.right.as_ref())?;
                match broadcast_shapes(vec!(left_shape, right_shape)) {
                    Some(shape) => Some(shape),
                    None => {
                        self.errs.push(
                            ValidationError::new(
                                format!("cannot broadcast operands together. lhs {} and rhs {}", left_shape, right_shape),
                                ast.span
                            )
                        );
                        None
                    }
                }
            },
            AstKind::Monop(monop) => {
                let dim = self.get_shape(monop.child.as_ref())?;
                Some(dim)
            },
            AstKind::Call(call) => {
                let shapes = call.args.iter().map(|c| self.get_shape(c.kind.as_call_arg().unwrap())).collect(); 
                match broadcast_shapes(shapes) {
                    Some(shape) => Some(shape),
                    None => {
                        self.errs.push(
                            ValidationError::new(
                                format!("cannot broadcast operands together. lhs {} and rhs {}", left_shape, right_shape),
                                ast.span
                            )
                        );
                        None
                    }
                }
            },
            AstKind::CallArg(arg) => todo!(),
            AstKind::Index(i) => todo!(),
            AstKind::Slice(s) => todo!(),
            AstKind::Number(n) => todo!(),
            AstKind::Integer(i) => todo!(),
            AstKind::Name(name) => todo!(),
            _ => panic!("unrecognised ast node {}", ast.kind)
        }
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'s> {
    pub name: &'s str,
    pub lhs: Array<'s>,
    pub rhs: Array<'s>,
    pub out: Array<'s>,
    pub in_defns: Vec<Array<'s>>,
    pub out_defns: Vec<Array<'s>>,
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
        write!(f, "in {{\n  {}\n}}\n", inputs_str.join("\n  "))
        .and_then(|_|
            self.in_defns.iter().fold(Ok(()), |result, array| {
                result.and_then(|_| writeln!(f, "{}", array))
            })
        )
        .and_then(|_|
            write!(f, "u {{\n  {}\n}}\n", states_str.join("\n  "))
        ).and_then(|_|
            write!(f, "{}\n", self.lhs)
        ).and_then(|_|
            write!(f, "{}\n", self.rhs)
        ).and_then(|_|
            self.out_defns.iter().fold(Ok(()), |result, array| {
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
            lhs: Array::new("F"),
            rhs: Array::new("G"),
            out: Array::new("out"),
            in_defns: Vec::new(),
            out_defns: Vec::new(),
            inputs: Vec::new(),
            states: Vec::new(),
        }
    }


    fn build_states(array: &ast::Array, env: &mut Env) -> Vec<State<'s>> {
        let ret = Vec::new();
        assert_eq!(array.name == "u");
        for a in array.elmts {
            if let Some(dim) = env.get_shape(a) {
                ret.push(ArrayElmt{ bounds: (0, dim), expr: a.clone() })  
            }
        }
        ret
    }

    pub fn build(name: &'s str, ast: &'s Vec<Box<Ast<'s>>>) -> Result<Self, ValidationErrors> {
        let mut env = Env::new();
        let ret = Self::new(name);
        let read_state= false;
        let read_F = false;
        let read_G = false;
        let read_out = false;
        for (i, array_ast) in ast.iter().enumerate() {
            match array_ast.kind.as_array() {
                None => errors.push(ValidationError::new("not an array".to_string(), array_ast.span)),
                Some(array) => {
                    let span = array_ast.span;
                    // first array must be in
                    if i == 0 &&  array.name != "in" {
                        errors.push(ValidationError::new("first array must be 'in'".to_string(), span));
                    }
                    match array.name {
                        "in" => {
                            self.build_inputs(array, ret.input, env);
                        }
                        "u" => {
                            read_state = true;
                            self.build_states(array, env);
                        },
                        "F" => {
                            read_F = true;
                            let built_array_elmts = self.build_array(array, errors);
                            ret.lhs.elmts.extend(built_array_elmts);
                        },
                        "G" => {
                            read_G = true;
                            let built_array_elmts = self.build_array(array, errors);
                            ret.rhs.elmts.extend(built_array_elmts);
                        },
                        "out" => {
                            read_out = true;
                            let built_array_elmts = self.build_array(array, errors);
                            ret.out.elmts.extend(built_array_elmts);
                        },
                        name => {
                            let built_array_elmts = self.build_array(array, errors);
                            let dependent_on_state = true;
                            let new_array = Array { name, elmts: built_array_elmt };
                            if dependent_on_state {
                                ret.in_defns.push(new_array);
                            } else {
                                ret.out_defns.push(new_array);
                            }
                        }
                    }
                },
            }

            
            

        }
        Err(errors)
    }
    pub fn len_state(&self) -> u32 {
        self.states.iter().fold(0, |sum, i| sum + i.get_shape())
    }
    pub fn len_inputs(&self) -> u32 {
        self.inputs.iter().fold(0, |sum, i| sum + i.get_shape())
    }
    pub fn len_output(&self) -> u32 {
        self.out.get_shape()
    }
    pub fn get_init_state(&self) -> (Array<'s>, Array<'s>) {
        let alg_init = Ast {
            kind: AstKind::Number(0.),
            span: None,
        };
        (
            Array {
                name: "u0",
                elmts: self.states.iter().map(
                    |s| ArrayElmt{ bounds: s.bounds, expr: match &s.init { Some(eq) => eq.clone(), None => alg_init.clone(), } } 
                ).collect(),
            },
            Array {
                name: "dotu0",
                elmts: self.states.iter().map(
                    |s| ArrayElmt{ bounds: s.bounds, expr: alg_init.clone() } 
                ).collect(),
            }
        )
    }
    fn state_to_elmt(state_cell: &Rc<RefCell<Variable<'s>>>) -> (ArrayElmt<'s>, ArrayElmt<'s>) {
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
            ArrayElmt{ expr: Ast { kind: f_astkind, span: ast_eqn.span }, bounds: (0, u32::try_from(state.dim).unwrap()) },
            ArrayElmt{ expr: Ast { kind: g_astkind, span: ast_eqn.span }, bounds: (0, u32::try_from(state.dim).unwrap()) },
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
    fn idfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Array<'s> {
        let defn = defn_cell.borrow();
        assert!(!defn.is_dependent_on_state());
        Array {
            name: defn.name,
            elmts: vec![ArrayElmt {expr: defn.expression.as_ref().unwrap().clone(), bounds: (0, u32::try_from(defn.dim).unwrap()) }],
        }
    }
    fn odfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Array<'s> {
        let defn = defn_cell.borrow();
        assert!(defn.is_dependent_on_state());
        Array {
            name: defn.name,
            elmts: vec![ArrayElmt {expr: defn.expression.as_ref().unwrap().clone(), bounds: (0, u32::try_from(defn.dim).unwrap())}],
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
    fn output_to_elmt(output_cell: &Rc<RefCell<Variable<'s>>>) -> ArrayElmt<'s> {
        let output = output_cell.borrow();
        ArrayElmt {
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
        
        let mut out_array_elmts: Vec<ArrayElmt> = chain(time_varying_unknowns.iter(), model.definitions.iter())
            .map(DiscreteModel::output_to_elmt).collect();
        let mut curr_index = 0;
        for elmt in out_array_elmts.iter_mut() {
            elmt.bounds.0 += curr_index;
            elmt.bounds.1 += curr_index;
            curr_index = elmt.bounds.1;
        }
        let out_array = Array {
            name: "out",
            elmts: out_array_elmts,
        };
        
        let mut f_elmts: Vec<ArrayElmt> = Vec::new();
        let mut g_elmts: Vec<ArrayElmt> = Vec::new();
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
        let lhs = Array {
            name: "F",
            elmts: f_elmts,
        };
        let rhs = Array {
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
 