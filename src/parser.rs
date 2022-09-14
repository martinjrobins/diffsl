use crate::pest::Parser;
use crate::MsParser;
use crate::Rule;
use pest::error::Error;
use pest::iterators::Pair;
use std::rc::Rc;

use crate::ast::Ast;
use crate::ast::AstKind;

//sign       = @{ ("-"|"+")? }
//factor_op  = @{ "*"|"/" }
fn parse_sign(pair: Pair<Rule>) -> char {
    print!("pair '{}'\n", pair.as_str());
    *pair
        .as_str()
        .chars()
        .collect::<Vec<char>>()
        .first()
        .unwrap()
}

//name       = @{ 'a'..'z' ~ ("_" | 'a'..'z' | 'A'..'Z' | '0'..'9')* }
//domain_name = @{ 'A'..'Z' ~ ('a'..'z' | 'A'..'Z' | '0'..'9')* }
fn parse_name(pair: Pair<Rule>) -> &str {
    pair.as_str()
}

fn parse_value<'a>(pair: Pair<'a, Rule>) -> Ast<'a> {
    let pos_start = pair.as_span().start();
    let pos_end = pair.as_span().end();
    match pair.as_rule() {
        // name       = @{ 'a'..'z' ~ ("_" | 'a'..'z' | 'A'..'Z' | '0'..'9')* }
        // domain_name = @{ 'A'..'Z' ~ ('a'..'z' | 'A'..'Z' | '0'..'9')* }
        Rule::name | Rule::domain_name => Ast {
            kind: AstKind::Name(pair.as_str()),
            pos_start,
            pos_end,
        },

        // integer    = @{ ('0'..'9')+ }
        // real       = @{ ( ('0'..'9')+ ~ "." ~ ('0'..'9')+ ) | integer }
        Rule::integer | Rule::real => Ast {
            kind: AstKind::Name(pair.as_str()),
            pos_start,
            pos_end,
        },

        // model = { "model" ~ name ~ "(" ~ unknown? ~ ("," ~ unknown)* ~ ")" ~ "{" ~ statement* ~ "}" }
        Rule::model => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let unknowns = inner
                .by_ref()
                .take_while(|pair| pair.as_rule() == Rule::unknown)
                .map(parse_value)
                .map(Rc::new)
                .collect();
            let statements = inner.map(parse_value).map(Rc::new).collect();
            Ast {
                kind: AstKind::Model {
                    name,
                    unknowns,
                    statements,
                    info: None,
                },
                pos_start,
                pos_end,
            }
        }
        // definition = { "let" ~ name ~ "=" ~ expression }
        Rule::definition => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let rhs = Rc::new(parse_value(inner.next().unwrap()));
            Ast {
                kind: AstKind::Definition { name, rhs },
                pos_start,
                pos_end,
            }
        }

        // range      = { "[" ~ real ~ "..." ~ real ~ "]" }
        Rule::range => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::Range {
                    lower: inner.next().unwrap().as_str().parse().unwrap(),
                    upper: inner.next().unwrap().as_str().parse().unwrap(),
                },
                pos_start,
                pos_end,
            }
        }

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
                inner.next().unwrap().into_inner().map(parse_name).collect()
            } else {
                Vec::new()
            };
            let codomain = if inner.peek().is_some() {
                Some(Rc::new(parse_value(inner.next().unwrap())))
            } else {
                None
            };
            Ast {
                kind: AstKind::Unknown {
                    name,
                    dependents,
                    codomain,
                },
                pos_start,
                pos_end,
            }
        }
        //statement  = { definition | submodel | rate_equation | equation }
        Rule::statement => parse_value(pair.into_inner().next().unwrap()),

        //call_arg   = { (name ~ "=")? ~ expression }
        Rule::call_arg => {
            let mut inner = pair.into_inner();
            let name = if inner.peek().unwrap().as_rule() == Rule::name {
                Some(parse_name(inner.next().unwrap()))
            } else {
                None
            };
            Ast {
                kind: AstKind::CallArg {
                    name,
                    expression: Rc::new(parse_value(inner.next().unwrap())),
                },
                pos_start,
                pos_end,
            }
        }

        //call       = { name ~ "(" ~ call_arg ~ ("," ~ call_arg )* ~ ")" }
        Rule::call => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::Call {
                    fn_name: parse_name(inner.next().unwrap()),
                    args: inner.map(parse_value).collect(),
                },
                pos_start,
                pos_end,
            }
        }

        //submodel   = { "use" ~ call ~ ("as" ~ name)? }
        Rule::submodel => {
            // TODO: is there a better way of destructuring this?
            let mut inner = pair.into_inner();
            let (name, args) = if let Ast {
                kind: AstKind::Call { fn_name, args },
                pos_start,
                pos_end,
            } = parse_value(inner.next().unwrap())
            {
                (fn_name, args)
            } else {
                unreachable!()
            };
            let local_name = if inner.peek().is_some() {
                parse_name(inner.next().unwrap())
            } else {
                name.clone()
            };
            Ast {
                kind: AstKind::Submodel {
                    name,
                    local_name,
                    args,
                },
                pos_start,
                pos_end,
            }
        }

        //rate_equation = { "dot" ~ "(" ~ name ~ ")" ~ "+=" ~ expression }
        Rule::rate_equation => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::RateEquation {
                    name: parse_name(inner.next().unwrap()),
                    rhs: Rc::new(parse_value(inner.next().unwrap())),
                },
                pos_start,
                pos_end,
            }
        }

        //equation   = { expression ~ "=" ~ expression }
        Rule::equation => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::Equation {
                    lhs: Rc::new(parse_value(inner.next().unwrap())),
                    rhs: Rc::new(parse_value(inner.next().unwrap())),
                },
                pos_start,
                pos_end,
            }
        }

        //expression = { sign? ~ term ~ (term_op ~ term)* }
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
                head_term.kind = AstKind::Binop {
                    op: term_op,
                    left: Rc::new(head_term),
                    right: Rc::new(rhs_term),
                };
            }
            if sign.is_some() {
                Ast {
                    kind: AstKind::Monop {
                        op: sign.unwrap(),
                        child: Rc::new(head_term),
                    },
                    pos_start,
                    pos_end,
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
                head_factor = Ast {
                    kind: AstKind::Binop {
                        op: factor_op,
                        left: Rc::new(head_factor),
                        right: Rc::new(rhs_factor),
                    },
                    pos_start,
                    pos_end,
                };
            }
            head_factor
        }

        // factor     = { call | name | real | integer | "(" ~ expression ~ ")" }
        Rule::factor => parse_value(pair.into_inner().next().unwrap()),

        _ => unreachable!("{:?}", pair.to_string()),
    }
}

pub fn parse_string(text: &str) -> Result<Vec<Ast>, Error<Rule>> {
    let main = MsParser::parse(Rule::main, &text)?.next().unwrap();
    let models = main
        .into_inner()
        .take_while(|pair| pair.as_rule() != Rule::EOI)
        .map(parse_value)
        .collect();
    return Ok(models);
}

#[cfg(test)]
mod tests {
    use super::parse_string;
    use crate::ast::AstKind;

    #[test]
    fn empty_model() {
        const TEXT: &str = "model test() {}";
        let models = parse_string(TEXT).unwrap();
        assert_eq!(models.len(), 1);
        if let AstKind::Model {name, unknowns, statements, info } = models[0].kind {
            assert_eq!(name, "test");
            assert!(unknowns.is_empty());
            assert!(statements.is_empty());
        } else {
            panic!("should be of kind AstKind::Model");
        }
    }

    #[test]
    fn two_models() {
        let text = "
        model capacitor( i(t), v(t), c -> NonNegative) {
            i = c * dot(v)
        }
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        ";
        let models = parse_string(text).unwrap();
        assert_eq!(models.len(), 2);

        if let AstKind::Model {
            name,
            unknowns,
            statements,
            info,
        } = models[0].kind
        {
            assert_eq!(name, "capacitor");
            assert_eq!(unknowns.len(), 3);
            if let AstKind::Unknown { name, dependents, codomain } = unknowns[0].kind {
                assert_eq!(name, "i");
                assert_eq!(dependents.len(), 1);
                assert!(codomain.is_none());
            } else {
                panic!("should be unknown");
            }
            if let AstKind::Unknown { name, dependents, codomain } = unknowns[1].kind {
                assert_eq!(name, "v");
                assert_eq!(dependents.len(), 1);
                assert!(codomain.is_none());
            } else {
                panic!("should be unknown");
            }
            if let AstKind::Unknown { name, dependents, codomain } = unknowns[2].kind {
                assert_eq!(name, "c");
                assert_eq!(dependents.len(), 0);
                assert!(codomain.is_some());
            } else {
                panic!("should be unknown");
            }
            assert_eq!(statements.len(), 1);
            if let AstKind::Equation { lhs, rhs } = statements[0].kind {
                assert!(matches!(lhs.kind, AstKind::Name(name) if name == "i"));
                assert!(matches!(rhs.kind, AstKind::Binop{op, left: _, right: _} if op == '*'));
            } else {
                assert!(false, "not an equation")
            }
        } else {
            assert!(false, "not a model");
        }
    }

    #[test]
    fn rate_equation() {
        let text = "
        model diffusion( x -> Omega, d -> NonNegative, y(x) ) { 
            dot(y) = d * div(grad(y, x), x) 
        }
        ";
        let models = parse_string(text).unwrap();
        assert_eq!(models.len(), 1);

        if let AstKind::Model {
            name,
            unknowns,
            statements,
            info,
        } = models[0].kind
        {
            assert_eq!(name, "diffusion");
            assert_eq!(unknowns.len(), 3);
            assert_eq!(statements.len(), 1);
            if let AstKind::RateEquation { name, rhs } = statements[0].kind {
                assert_eq!(name, "y");
                assert!(matches!(rhs.kind, AstKind::Binop{op, left: _, right: _} if op == '*'));
            } else {
                assert!(false, "not a rate equation")
            }
        } else {
            assert!(false, "not a model");
        }
    }

    #[test]
    fn submodel_and_let() {
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
        assert_eq!(models.len(), 2);

        if let AstKind::Model {
            name,
            unknowns,
            statements,
            info,
        } = models[1].kind
        {
            assert_eq!(name, "circuit");
            assert_eq!(unknowns.len(), 3);
            assert_eq!(statements.len(), 2);
            if let AstKind::Definition { name, rhs } = statements[0].kind {
                assert_eq!(name, "inputVoltage");
                assert!(
                    matches!(rhs.kind, AstKind::Call{fn_name, args} if fn_name == "sin" && args.len() == 1)
                );
            } else {
                assert!(false, "not an definition")
            }
            if let AstKind::Submodel {
                name,
                local_name,
                args,
            } = statements[1].kind
            {
                assert_eq!(name, "resistor");
                assert_eq!(local_name, "resistor");
                assert_eq!(args.len(), 1);
                if let AstKind::CallArg { name, expression } = args[0].kind {
                    assert_eq!(name.unwrap(), "v");
                    assert!(matches!(expression.kind, AstKind::Name(name) if name == "inputVoltage"));
                } else {
                    unreachable!("not a call arg")
                }
            } else {
                assert!(false, "not an definition")
            }
        } else {
            assert!(false, "not a model");
        }
    }
}
