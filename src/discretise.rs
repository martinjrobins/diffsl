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
pub struct Array<'s> {
    name: &'s str,
    elmts: Vec<Ast<'s>>,
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
    dim: usize,
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
    dim: usize,
    init: Ast<'s>,
}

impl<'s> fmt::Display for State<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}", self.name, self.init)
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'s> {
    arrays: Vec<Array<'s>>,
    inputs: Vec<Input<'s>>,
    state: Vec<State<'s>>,
}

impl<'s, 'a> fmt::Display for DiscreteModel<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inputs_str: Vec<String> = self.inputs.iter().map(|i| i.to_string()).collect();
        let states_str: Vec<String> = self.state.iter().map(|i| i.to_string()).collect();
        let result = write!(f, "in {{\n  {}\n}}\n", inputs_str.join("\n  ")).and_then(|_|
            write!(f, "u {{\n  {}\n}}\n", states_str.join("\n  "))
        );
        self.arrays.iter().fold(result, |result, array| {
            result.and_then(|_| writeln!(f, "{}", array))
        })
    }
}

impl<'s> DiscreteModel<'s> {
    fn state_to_elmt(state_cell: &Rc<RefCell<Variable<'s>>>) -> (Ast<'s>, Ast<'s>) {
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
            Ast { kind: f_astkind, span: ast_eqn.span },
            Ast { kind: g_astkind, span: ast_eqn.span }
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
        State { name: state.name, dim: state.dim, init }
    }
    fn state_to_array(state_cell: &Rc<RefCell<Variable<'s>>>, indices: (usize, usize)) -> Array<'s> {
        let state = state_cell.borrow();
        let span = if let Some(eqn) = &state.equation {
            eqn.span
        } else {
            panic!("state var should have an equation")
        };
        let state_name = Ast {
            kind: AstKind::new_name("u"),
            span: span,
        };
        let range = if indices.0 == indices.1 - 1 {
            Ast {
                kind: AstKind::new_int(indices.0 as i64),
                span: span,
            }
        } else {
            Ast {
                kind: AstKind::new_irange(indices),
                span: span,
            }
        };
        let eqn = Ast {
            kind: AstKind::new_index(state_name, range),
            span: span,
        };
        Array {
            name: state.name,
            elmts: vec![eqn],
        }
    }
    fn dfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Option<Array<'s>> {
        let defn = defn_cell.borrow();
        if defn.is_dependent_on_state() {
            None
        } else {
            Some(Array {
                name: defn.name,
                elmts: vec![defn.expression.as_ref().unwrap().clone()],
            })
        }
    }
    fn state_to_input(input_cell: &Rc<RefCell<Variable<'s>>>) -> Input<'s> {
        let input = input_cell.borrow();
        assert!(input.is_independent());
        assert!(!input.is_time_dependent());
        Input {
            name: input.name,
            dim: input.dim,
            bounds: input.bounds,
        }
    }
    fn output_to_elmt(output_cell: &Rc<RefCell<Variable<'s>>>) -> Option<Ast<'s>> {
        let output = output_cell.borrow();
        if output.is_definition() {
            if output.is_dependent_on_state() {
                Some(output.expression.as_ref().unwrap().clone())
            } else {
                None
            }
        } else {
            Some(Ast {
                kind: AstKind::new_name(output.name),
                span: if output.is_definition() { 
                    output.expression.as_ref().unwrap().span 
                } else if output.has_equation() { 
                    output.equation.as_ref().unwrap().span 
                } else { None } 
            })
        }
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

        let (start_indices, end_indices): (Vec<usize>, Vec<usize>) = states.iter().scan((0, 0), |s, v|  { 
            *s = (s.1, s.1 + v.borrow().dim); Some(*s) 
        }).unzip();
        let (f_elmts, g_elmts) = states
            .iter()
            .map(DiscreteModel::state_to_elmt)
            .unzip();
        let init_states = states
            .iter()
            .map(DiscreteModel::state_to_u0)
            .collect();

        let mut arrays: Vec<Array> = Vec::new(); 
        let inputs: Vec<Input> = inputs.iter().map(DiscreteModel::state_to_input).collect();
        arrays.extend(
            defns 
            .iter()
            .filter_map(DiscreteModel::dfn_to_array)
        );
        arrays.push(Array {
            name: "F",
            elmts: f_elmts,
        });
        arrays.push(Array {
            name: "G",
            elmts: g_elmts,
        });
        arrays.push(out_array);
        DiscreteModel {
            inputs,
            arrays,
            state: init_states,
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
        assert_eq!(discrete.arrays.len(), 4);
        assert_eq!(discrete.arrays[0].name, "inputVoltage");
        assert_eq!(discrete.arrays[0].elmts.len(), 1);
        assert_eq!(discrete.arrays[1].name, "F");
        assert_eq!(discrete.arrays[1].elmts.len(), 1);
        assert_eq!(discrete.arrays[2].name, "G");
        assert_eq!(discrete.arrays[2].elmts.len(), 1);
        assert_eq!(discrete.arrays[3].name, "out");
        assert_eq!(discrete.arrays[3].elmts.len(), 2);
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
 