use core::panic;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Binop;
use crate::builder::ModelInfo;
use crate::builder::Variable;

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct Array<'s> {
    pub name: &'s str,
    pub elmts: Vec<Ast<'s>>,
}

impl<'s> fmt::Display for Array<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let elmts_str: Vec<String> = self.elmts.iter().map(|e| e.to_string()).collect();
        write!(f, "{} {{\n  {}\n}}", self.name, elmts_str.join(",\n"))
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'s> {
    pub arrays: Vec<Array<'s>>,
    pub n_states: usize,
}

impl<'s, 'a> fmt::Display for DiscreteModel<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.arrays.iter().fold(Ok(()), |result, array| {
            result.and_then(|_| writeln!(f, "{}", array))
        })
    }
}

impl<'s> DiscreteModel<'s> {
    fn state_to_elmt(state: &Rc<RefCell<Variable<'s>>>) -> (Ast<'s>, Ast<'s>) {
        let eqn = if let Some(eqn) = &state.borrow().equation {
            eqn.clone()
        } else {
            panic!("state var should have an equation")
        };
        let (f_astkind, g_astkind) = match eqn.kind {
            AstKind::RateEquation(eqn) => (
                AstKind::Name("dudt"),
                eqn.rhs.kind,
            ),
            AstKind::Equation(eqn) => (
                AstKind::Number(0.0),
                AstKind::Binop(Binop {
                    op: '-',
                    left: eqn.rhs,
                    right: eqn.lhs,
                }),
            ),
            _ => panic!("equation for state var should be rate eqn or standard eqn"),
        };
        (
            Ast { kind: f_astkind, span: eqn.span },
            Ast { kind: g_astkind, span: eqn.span },
        )
    }
    fn dfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Array<'s> {
        let defn = defn_cell.borrow();
        Array {
            name: defn.name,
            elmts: vec![defn.expression.as_ref().unwrap().clone()],
        }
        
    }
    pub fn from(model: ModelInfo<'s>) -> DiscreteModel<'s> {
        let (_inputs, time_varying): (Vec<_>, Vec<_>) = model
            .variables
            .into_iter()
            .map(|(_name, var)| var)
            .partition(|var| !var.borrow().is_time_dependent());
        let (states, defns): (Vec<_>, Vec<_>) = time_varying
            .into_iter()
            .partition(|v| v.borrow().is_state());
        let n_states = states.iter().fold(0, |s, v| s + v.borrow().dim);
        let (f_elmts, g_elmts) = states 
            .iter()
            .map(DiscreteModel::state_to_elmt)
            .unzip();
        let mut arrays: Vec<Array> = defns 
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        arrays.push(Array {
            name: "F",
            elmts: f_elmts,
        });
        arrays.push(Array {
            name: "G",
            elmts: g_elmts,
        });
        DiscreteModel {
            arrays,
            n_states,
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
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.output.len(), 0);
        let discrete = DiscreteModel::from(model_info);
        assert_eq!(discrete.arrays.len(), 3);
        assert_eq!(discrete.arrays[0].name, "inputVoltage");
        assert_eq!(discrete.arrays[0].elmts.len(), 1);
        assert_eq!(discrete.arrays[1].name, "F");
        assert_eq!(discrete.arrays[1].elmts.len(), 1);
        assert_eq!(discrete.arrays[2].name, "G");
        assert_eq!(discrete.arrays[2].elmts.len(), 1);
        println!("{}", discrete);
    }
}
 