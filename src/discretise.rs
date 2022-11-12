use std::fmt;
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
pub struct DiscreteModel<'s, 'a>
where
    'a: 's,
{
    pub arrays: Vec<Array<'s>>,
    pub inputs: Vec<Variable<'s, 'a>>,
    pub states: Vec<Variable<'s, 'a>>,
    pub n_states: usize,
}

impl<'s, 'a> fmt::Display for DiscreteModel<'s, 'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.arrays.iter().fold(Ok(()), |result, array| {
            result.and_then(|_| writeln!(f, "{}", array))
        })
    }
}

impl<'s, 'a> DiscreteModel<'s, 'a> {
    fn is_state_eqn(stmt: &Ast) -> bool {
        match stmt.kind {
            AstKind::RateEquation(_) | AstKind::Equation(_) => true,
            _ => false,
        }
    }
    fn state_eqn_to_elmt(stmt: Ast<'s>) -> (Ast<'s>, Ast<'s>) {
        let span = stmt.span;
        let (f_astkind, g_astkind) = match stmt.kind {
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
            _ => unreachable!(),
        };
        (
            Ast { kind: f_astkind, span },
            Ast { kind: g_astkind, span },
        )
    }
    fn dfn_eqn_to_array(stmt: Ast<'s>) -> Array<'s> {
        if let AstKind::Definition(defn) = stmt.kind {
            Array {
                name: defn.name,
                elmts: vec![*defn.rhs],
            }
        } else {
            panic!("var should be a definition")
        }
    }
    pub fn from(model: ModelInfo<'s, 'a>) -> DiscreteModel<'s, 'a> {
        let (inputs, time_varying): (Vec<_>, Vec<_>) = model
            .variables
            .into_iter()
            .partition(|v| v.constant);
        let (states, _defns): (Vec<_>, Vec<_>) = time_varying
            .into_iter()
            .partition(|v| v.state);

        let n_states = states.iter().fold(0, |s, v| s + v.dim);
        let (state_eqns, dfn_eqns) : (Vec<_>, Vec<_>) = model.stmts.into_iter().partition(DiscreteModel::is_state_eqn);
        let (f_elmts, g_elmts) = state_eqns 
            .into_iter()
            .map(DiscreteModel::state_eqn_to_elmt)
            .unzip();
        let mut arrays: Vec<Array> = dfn_eqns 
            .into_iter()
            .map(DiscreteModel::dfn_eqn_to_array)
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
            inputs,
            states,
            n_states,
        }
    }
}




#[cfg(test)]
mod tests {
use crate::{ast::Model, parser::parse_string, discretise::DiscreteModel, builder::ModelInfo};

    #[test]
    fn test_circuit_model() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i1(t), i2(t), i3(t)) {
            let inputVoltage = sin(t) 
            use resistor(v = inputVoltage)
        }
        ";
        let models = parse_string(text).unwrap();
        let models_ref: Vec<&Model> = models.iter().collect();
        let model_info = ModelInfo::build("circuit", &models_ref).unwrap();
        println!("{:#?}", model_info);
        let discrete = DiscreteModel::from(model_info);
        println!("{:?}", discrete);
        println!("{}", discrete);
    }
}
 