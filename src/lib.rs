extern crate pest;
#[macro_use]
extern crate pest_derive;

#[derive(Parser)]
#[grammar = "ms_grammar.pest"] // relative to src
struct MsParser;

use crate::pest::Parser;
use pest::error::Error;
use pest::iterators::Pair;

#[derive(Debug)]
pub enum Expr {
    Binop {
        op: char,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    Monop {
        op: char,
        child: Box<Expr>,
    },

    Call {
        fn_name: String,
        args: Vec<Expr>,
    },

    Number(f64),

    Variable(String),
}

#[derive(Debug)]
pub enum EquationType {
    Algebraic,
    RateEqn { variable: Variable },
}

#[derive(Debug)]
pub struct Equation {
    kind: EquationType,
    rhs: Box<Expr>,
}

#[derive(Debug)]
pub struct Domain {
    name: String,
    extents: (f64, f64),
}

#[derive(Debug)]
pub struct Variable {
    name: String,
    model: Model,
    domain: Domain,
    constant: bool,
    state: bool,
}

#[derive(Debug)]
pub struct Model {
    name: String,
    variables: Vec<Variable>,
    equations: Vec<Equation>,
    submodels: Vec<Model>,
}

impl Model {
    pub fn create(rule: Pair<Rule>) -> Model {
        // "model" ~ name ~ "(" ~ unknown? ~ ("," ~ unknown)* ~ ")" ~ "{" ~ statement* ~ "}" 
        let mut inner_rules = line.into_inner();
        let name = inner_rules.next().unwrap().as_str();
        let mut model = Model {
            name: name,
            variables: Vec::new(),
            equations: Vec::new(),
            submodels: Vec::new(),
        };
    }
}

fn parse_string(text: &str) -> Result<Vec<Model>, Error<Rule>> {
    let main = MsParser::parse(Rule::main, &text)?.next().unwrap();
    let models = main
        .into_inner()
        .filter(|m| m.as_rule() == Rule::model)
        .map(|m| match m.as_rule() {
            Rule::model => Model::create(m),
            _ => unreachable!(),
        });
    return Ok(models.collect());
}

#[cfg(test)]
mod tests {
    use crate::parse_string;
    use crate::pest::Parser;
    use crate::MsParser;
    use crate::Rule;
    use std::fs;

    const MS_FILENAMES: &[&str] = &["test_circuit.ms", "test_fishers.ms", "test_pk.ms"];

    const BASE_DIR: &str = "src";

    #[test]
    fn can_parse() {
        for filename in MS_FILENAMES {
            let unparsed_file =
                fs::read_to_string(BASE_DIR.to_owned() + "/" + filename).expect("cannot read file");
            let _list = MsParser::parse(Rule::main, &unparsed_file).expect("unsuccessful parse");
        }
    }

    #[test]
    fn parse_model() {
        const TEXT: &str = "model test() {}";
        let models = parse_string(TEXT).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test");
        assert_eq!(models[0].equations.len(), 0);
        assert_eq!(models[0].variables.len(), 0);
        assert_eq!(models[0].submodels.len(), 0);
    }
}
