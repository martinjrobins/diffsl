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
pub enum Ast {
    Model {
        name: String,
        unknowns: Vec<Ast>,
        statements: Vec<Ast>,
    },
    Unknown {
        name: String,
        dependents: Vec<String>,
        codomain: Option<Box<Ast>>,
    },

    Definition {
        name: String,
        rhs: Box<Ast>,
    },

    Submodel {
        name: String,
        local_name: String,
        args: Vec<Ast>,
    },

    Equation {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },

    RateEquation {
        name: String,
        rhs: Box<Ast>,
    },

    Range {
        lower: f64,
        upper: f64,
    },

    Binop {
        op: char,
        left: Box<Ast>,
        right: Box<Ast>,
    },

    Monop {
        op: char,
        child: Box<Ast>,
    },

    Call {
        fn_name: String,
        args: Vec<Ast>,
    },

    CallArg {
        name: String,
        expression: Box<Ast>,
    },

    Number(f64),

    Name(String),
}

pub fn print_ast(ast: &Ast) {
    match ast {
        Ast::Model { name, unknowns, statements } => {
            println!("Model ({})", name)
            println!("Unknowns ({})", name)
        },

    }
}

//sign       = @{ ("-"|"+")? }
//factor_op  = @{ "*"|"/" }
pub fn parse_sign(pair: Pair<Rule>) -> char {
    *pair.into_inner().next().unwrap().as_str().chars().collect::<Vec<char>>().first().unwrap()
}

//name       = @{ 'a'..'z' ~ ("_" | 'a'..'z' | 'A'..'Z' | '0'..'9')* }
//domain_name = @{ 'A'..'Z' ~ ('a'..'z' | 'A'..'Z' | '0'..'9')* }
pub fn parse_name(pair: Pair<Rule>) -> String {
    pair.into_inner().next().unwrap().as_str().to_string()
}

pub fn parse_value<'a>(pair: Pair<Rule>) -> Ast {
    match pair.as_rule() {
        // name       = @{ 'a'..'z' ~ ("_" | 'a'..'z' | 'A'..'Z' | '0'..'9')* }
        // domain_name = @{ 'A'..'Z' ~ ('a'..'z' | 'A'..'Z' | '0'..'9')* }
        Rule::name | Rule::domain_name => Ast::Name(pair.into_inner().next().unwrap().as_str().to_string()),

        // integer    = @{ ('0'..'9')+ }
        // real       = @{ ( ('0'..'9')+ ~ "." ~ ('0'..'9')+ ) | integer }
        Rule::integer | Rule::real => Ast::Name(pair.into_inner().next().unwrap().as_str().parse().unwrap()),

        // model = { "model" ~ name ~ "(" ~ unknown? ~ ("," ~ unknown)* ~ ")" ~ "{" ~ statement* ~ "}" }
        Rule::model => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let unknowns = inner.by_ref()
                .take_while(|pair| pair.as_rule() == Rule::unknown)
                .map(parse_value)
                .collect();
            let statements = inner.map(parse_value).collect();
            Ast::Model {
                name,
                unknowns,
                statements,
            }
        }
        // definition = { "let" ~ name ~ "=" ~ expression }
        Rule::definition => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let rhs = Box::new(parse_value(inner.next().unwrap()));
            Ast::Definition { name, rhs }
        }

        // range      = { "[" ~ real ~ "..." ~ real ~ "]" }
        Rule::range => {
            let mut inner = pair.into_inner();
            Ast::Range {
                lower: inner.next().unwrap().as_str().parse().unwrap(),
                upper: inner.next().unwrap().as_str().parse().unwrap(),
            }
        },

        // domain     = { range | domain_name }
        Rule::domain => parse_value(pair.into_inner().next().unwrap()),

        // codomain   = { "->" ~ domain }
        Rule::codomain => parse_value(pair.into_inner().next().unwrap()),

        //unknown    = { name ~ dependents? ~ codomain? }
        Rule::unknown => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            //dependents = { "(" ~ name ~ ("," ~ name )* ~ ")" }
            let dependents = if inner.peek().unwrap().as_rule() == Rule::dependents {
                inner
                    .next()
                    .unwrap()
                    .into_inner()
                    .map(parse_name)
                    .collect()
            } else {
                Vec::new()
            };
            let codomain = if inner.peek().is_some() {
                Some(Box::new(parse_value(inner.next().unwrap())))
            } else {
                None
            };
            Ast::Unknown {
                name,
                dependents,
                codomain,
            }
        }
        //statement  = { definition | submodel | rate_equation | equation }
        Rule::statement => parse_value(pair.into_inner().next().unwrap()),

        //call_arg   = { name ~ "=" ~ expression }
        Rule::call_arg => {
            let mut inner = pair.into_inner();
            Ast::CallArg {
                name: parse_name(inner.next().unwrap()),
                expression: Box::new(parse_value(inner.next().unwrap())),
            }
        },

        //call       = { name ~ "(" ~ call_arg ~ ("," ~ call_arg )* ~ ")" }
        Rule::call => {
            let mut inner = pair.into_inner();
            Ast::Call {
                fn_name: parse_name(inner.next().unwrap()),
                args: inner
                    .next()
                    .unwrap()
                    .into_inner()
                    .map(parse_value)
                    .collect(),
            }
        },

        //submodel   = { "use" ~ call ~ ("as" ~ name)? }
        Rule::submodel => {
            // TODO: is there a better way of destructuring this?
            let mut inner = pair.into_inner();
            let submodel = 
                if let Ast::Call { fn_name, args } = parse_value(inner.next().unwrap()) {
                    (fn_name, args)
                } else {
                    unreachable!()
                };
            let local_name = parse_name(inner.next().unwrap());
            Ast::Submodel {
                name: submodel.0,
                local_name,
                args: submodel.1
            }
        }

        //rate_equation = { "dot" ~ "(" ~ name ~ ")" ~ "+=" ~ expression }
        Rule::rate_equation => {
            let mut inner = pair.into_inner();
            Ast::RateEquation {
                name: parse_name(inner.next().unwrap()),
                rhs: Box::new(parse_value(inner.next().unwrap())),
            }
        },

        //equation   = { expression ~ "=" ~ expression }
        Rule::equation => {
            let mut inner = pair.into_inner();
            Ast::Equation {
                lhs: Box::new(parse_value(inner.next().unwrap())),
                rhs: Box::new(parse_value(inner.next().unwrap())),
            }
        },

        //expression = { sign ~ term ~ (term_op ~ term)* }
        Rule::expression => {
            let mut inner = pair.into_inner();
            let sign = if inner.peek().unwrap().as_rule() == Rule::sign {
                Some(parse_sign(inner.next().unwrap()))
            } else {
                None
            };
            let mut head_term = parse_value(inner.next().unwrap());
            while inner.peek().is_some() {
                //term_op    = @{ "-"|"+" }
                let term_op = parse_sign(inner.next().unwrap());
                let rhs_term = parse_value(inner.next().unwrap());
                head_term = Ast::Binop {
                    op: term_op,
                    left: Box::new(head_term),
                    right: Box::new(rhs_term),
                };
            }
            if sign.is_some() {
                Ast::Monop {
                    op: sign.unwrap(),
                    child: Box::new(head_term),
                }
            } else {
                head_term
            }
        }

        //term       = { factor ~ (factor_op ~ factor)* }
        Rule::term => {
            let mut inner = pair.into_inner();
            let mut head_factor = parse_value(inner.next().unwrap());
            while inner.peek().is_some() {
                let factor_op = parse_sign(inner.next().unwrap());
                let rhs_factor = parse_value(inner.next().unwrap());
                head_factor = Ast::Binop {
                    op: factor_op,
                    left: Box::new(head_factor),
                    right: Box::new(rhs_factor),
                };
            }
            head_factor
        }
        _ => unreachable!(),
    }
}

pub fn parse_string(text: &str) -> Result<Ast, Error<Rule>> {
    let main = MsParser::parse(Rule::main, &text)?.next().unwrap();
    let ast = parse_value(main);
    return Ok(ast);
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
