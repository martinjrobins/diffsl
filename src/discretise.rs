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
        write!(f, "{}..{}: {}", self.bounds.0, self.bounds.1, self.expr)
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
    pub name: &'s str,
    pub bounds: (u32, u32),
}

impl<'s> fmt::Display for Input<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dim = self.get_dim();
        if dim > 1 {
            write!(f, "{}^{}", self.name, dim)
        } else {
            write!(f, "{}", self.name)
        }.and_then(|_|
            write!(f, " -> [{}, {}]", self.bounds.0, self.bounds.1)
        )
    }
}

impl<'s> Input<'s> {
    pub fn get_dim(&self) -> u32 {
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
    pub fn get_dim(&self) -> u32 {
        self.bounds.1 - self.bounds.0
    }
    pub fn is_algebraic(&self) ->bool {
        self.init.is_none()
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
            inputs_str.push(format!("{}..{}: {}", input.bounds.0, input.bounds.1, input));
        }
        let mut states_str: Vec<String> = Vec::new();
        for state in self.states.iter() {
            states_str.push(format!("{}..{}: {}", state.bounds.0, state.bounds.1, state));
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
    pub fn len_state(&self) -> u32 {
        self.states.iter().fold(0, |sum, i| sum + i.get_dim())
    }
    pub fn len_inputs(&self) -> u32 {
        self.inputs.iter().fold(0, |sum, i| sum + i.get_dim())
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
            bounds: (0, u32::try_from(input.dim).unwrap()),
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
        let (parameters, time_varying): (Vec<_>, Vec<_>) = model
            .variables
            .into_iter()
            .map(|(_name, var)| var)
            .partition(|var| !var.borrow().is_time_dependent());
        
        let mut curr_index = 0;
        let mut out_array_elmts: Vec<ArrayElmt> = Vec::new();
        for out in time_varying.iter() {
            if out.borrow().is_time() {
                continue
            }
            if let Some(mut elmt) = DiscreteModel::output_to_elmt(out) {
                elmt.bounds.0 += curr_index;
                elmt.bounds.1 += curr_index;
                curr_index = elmt.bounds.1;
                out_array_elmts.push(elmt);
            }
        }
        let out_array = Array {
            name: "out",
            elmts: out_array_elmts,
        };
        let (states, defns): (Vec<_>, Vec<_>) = time_varying
            .into_iter()
            .partition(|v| v.borrow().is_state());
        let (odefns, idefns): (Vec<_>, Vec<_>) =  
            defns 
            .iter()
            .filter(|d| !d.borrow().is_time())
            .partition(|v| v.borrow().is_dependent_on_state());

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
        for input in parameters.iter() {
            let mut inp = DiscreteModel::state_to_input(input);
            inp.bounds.0 += curr_index;
            inp.bounds.1 += curr_index;
            curr_index = inp.bounds.1;
            inputs.push(inp);
        }
        
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
 