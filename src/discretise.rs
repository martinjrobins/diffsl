use core::panic;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::ast::Ast;
use crate::ast::AstKind;
use crate::builder::ModelInfo;
use crate::builder::Variable;

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct ArrayElmt<'s> {
    pub bounds: (u32, u32),
    pub expr: Ast<'s>,
}

impl<'s> ArrayElmt<'s> {
    pub fn get_dim(&self) -> u32 {
        self.bounds.1 - self.bounds.0
    }
}

impl<'s> fmt::Display for ArrayElmt<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{} -> {}", self.bounds.0, self.bounds.1, self.expr)
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct Array<'s> {
    pub name: &'s str,
    pub elmts: Vec<ArrayElmt<'s>>,
}

impl<'s> Array<'s> {
    pub fn get_dim(&self) -> u32 {
        self.elmts.iter().fold(0, |sum, e| sum + e.get_dim())
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
    name: &'s str,
    dim: u32,
    bounds: (f64, f64),
}

impl<'s> fmt::Display for Input<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.dim > 1 {
            write!(f, "{}^{}", self.name, self.dim)
        } else {
            write!(f, "{}", self.name)
        }.and_then(|_|
            write!(f, " -> [{}, {}]", self.bounds.0, self.bounds.1)
        )
    }
}

#[derive(Debug)]
// the p[i] in F(t, p, u, u_dot) = G(t, p, u)
pub struct State<'s> {
    name: &'s str,
    dim: u32,
    init: Ast<'s>,
}

impl<'s> fmt::Display for State<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}", self.name, self.init)
    }
}

impl<'s> State<'s> {
    pub fn is_algebraic(&self) ->bool {
        match self.init.kind {
            AstKind::Number(value) => f64::is_nan(value),
            _ => false,
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
        let inputs_str: Vec<String> = self.inputs.iter().map(|i| i.to_string()).collect();
        let states_str: Vec<String> = self.states.iter().map(|i| i.to_string()).collect();
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
    pub fn len_state(&self) -> u32 {
        self.states.iter().fold(0, |sum, i| sum + i.dim)
    }
    pub fn len_inputs(&self) -> u32 {
        self.inputs.iter().fold(0, |sum, i| sum + i.dim)
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
        let span = if let Some(eqn) = &state.equation {
            eqn.span
        } else {
            panic!("state var should have an equation")
        };
        let init = if state.has_initial_condition() {
            state.init_conditions[0].equation.clone()
        } else {
            Ast {
                kind: AstKind::new_num(f64::NAN),
                span,
            }
        };
        State { name: state.name, dim: u32::try_from(state.dim).expect("cannot convert usize -> u32"), init }
    }
    fn idfn_to_array(defn_cell: &&Rc<RefCell<Variable<'s>>>) -> Array<'s> {
        let defn = defn_cell.borrow();
        assert!(!defn.is_dependent_on_state());
        Array {
            name: defn.name,
            elmts: vec![ArrayElmt {expr: defn.expression.as_ref().unwrap().clone(), bounds: (0, u32::try_from(defn.dim).unwrap()) }],
        }
    }
    fn odfn_to_array(defn_cell: &&Rc<RefCell<Variable<'s>>>) -> Array<'s> {
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
            dim: u32::try_from(input.dim).unwrap(),
            bounds: input.bounds,
        }
    }
    fn output_to_elmt(output_cell: &Rc<RefCell<Variable<'s>>>) -> Option<ArrayElmt<'s>> {
        let output = output_cell.borrow();
        Some(ArrayElmt {
            expr: Ast {
                kind: AstKind::new_name(output.name),
                span: if output.is_definition() { 
                    output.expression.as_ref().unwrap().span 
                } else if output.has_equation() { 
                    output.equation.as_ref().unwrap().span 
                } else { None } 
            },
            bounds: (0, u32::try_from(output.dim).unwrap())
    })
    }
    pub fn from(model: ModelInfo<'s>) -> DiscreteModel<'s> {
        let (inputs, time_varying): (Vec<_>, Vec<_>) = model
            .variables
            .into_iter()
            .map(|(_name, var)| var)
            .partition(|var| !var.borrow().is_time_dependent());
        let out_array = Array {
            name: "out",
            elmts: time_varying.iter().filter_map(DiscreteModel::output_to_elmt).collect(),
        };
        let (states, defns): (Vec<_>, Vec<_>) = time_varying
            .into_iter()
            .partition(|v| v.borrow().is_state());
        let (odefns, idefns): (Vec<_>, Vec<_>) =  
            defns 
            .iter()
            .partition(|v| v.borrow().is_dependent_on_state());

        let (f_elmts, g_elmts) = states
            .iter()
            .map(DiscreteModel::state_to_elmt)
            .unzip();
        let init_states = states
            .iter()
            .map(DiscreteModel::state_to_u0)
            .collect();

        let inputs: Vec<Input> = inputs.iter().map(DiscreteModel::state_to_input).collect();
        let in_defns = idefns.iter().map(DiscreteModel::idfn_to_array).collect();
        let out_defns = odefns.iter().map(DiscreteModel::odfn_to_array).collect();
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
use crate::{parser::parse_string, discretise::DiscreteModel, builder::ModelInfo};

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
        assert_eq!(model_info.output.len(), 0);
        let discrete = DiscreteModel::from(model_info);
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
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), z(t) ) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
            z = 2 * y
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.output.len(), 0);
        let discrete = DiscreteModel::from(model_info);
        println!("{}", discrete);
    }
}
 