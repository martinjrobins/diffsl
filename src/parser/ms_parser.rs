#[derive(Parser)]
#[grammar = "parser/ms_grammar.pest"] // relative to src
pub struct MsParser;

use crate::ast::StringSpan;

use pest::Parser;
use pest::error::Error;
use pest::iterators::Pair;
use std::boxed::Box;

use crate::ast;
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

fn parse_value<'a, 'b>(pair: Pair<'a, Rule>) -> Ast<'a> {
    let span = Some(StringSpan {
        pos_start: pair.as_span().start(),
        pos_end: pair.as_span().end(),
    });
    match pair.as_rule() {
        // name       = @{ 'a'..'z' ~ ("_" | 'a'..'z' | 'A'..'Z' | '0'..'9')* }
        // domain_name = @{ 'A'..'Z' ~ ('a'..'z' | 'A'..'Z' | '0'..'9')* }
        Rule::name | Rule::domain_name => Ast {
            kind: AstKind::Name(pair.as_str()),
            span,
        },

        // integer    = @{ ('0'..'9')+ }
        // real       = @{ ( ('0'..'9')+ ~ "." ~ ('0'..'9')+ ) | integer }
        Rule::integer | Rule::real => Ast {
            kind: AstKind::Number(pair.as_str().parse().unwrap()),
            span,
        },

        // model = { "model" ~ name ~ "(" ~ unknown? ~ ("," ~ unknown)* ~ ")" ~ "{" ~ statement* ~ "}" }
        Rule::model => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let mut unknowns: Vec<Box<Ast>> = Vec::new();
            while inner.peek().is_some() && inner.peek().unwrap().as_rule() == Rule::unknown {
                unknowns.push(Box::new(parse_value(inner.next().unwrap())));
            }
            let statements = inner.map(parse_value).map(Box::new).collect();
            Ast {
                kind: AstKind::Model(ast::Model {
                    name,
                    unknowns,
                    statements,
                }),
                span,
            }
        }
        // definition = { "let" ~ name ~ "=" ~ expression }
        Rule::definition => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let rhs = Box::new(parse_value(inner.next().unwrap()));
            Ast {
                kind: AstKind::Definition(ast::Definition { name, rhs }),
                span,
            }
        }

        // range      = { "[" ~ real ~ "..." ~ real ~ "]" }
        Rule::range => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::Range(ast::Range {
                    lower: inner.next().unwrap().as_str().parse().unwrap(),
                    upper: inner.next().unwrap().as_str().parse().unwrap(),
                }),
                span,
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
            let dependents = if inner.peek().is_some() && inner.peek().unwrap().as_rule() == Rule::dependents {
                inner.next().unwrap().into_inner().map(parse_name).collect()
            } else {
                Vec::new()
            };
            let codomain = if inner.peek().is_some() {
                Some(Box::new(parse_value(inner.next().unwrap())))
            } else {
                None
            };
            Ast {
                kind: AstKind::Unknown(ast::Unknown {
                    name,
                    dependents,
                    codomain,
                }),
                span,
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
                kind: AstKind::CallArg(ast::CallArg {
                    name,
                    expression: Box::new(parse_value(inner.next().unwrap())),
                }),
                span,
            }
        }

        //call       = { name ~ "(" ~ call_arg ~ ("," ~ call_arg )* ~ ")" }
        Rule::call => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::Call(ast::Call {
                    fn_name: parse_name(inner.next().unwrap()),
                    args: inner.map(parse_value).map(Box::new).collect(),
                }),
                span,
            }
        }

        //submodel   = { "use" ~ call ~ ("as" ~ name)? }
        Rule::submodel => {
            // TODO: is there a better way of destructuring this?
            let mut inner = pair.into_inner();
            let (name, args) = if let Ast {
                kind: AstKind::Call(ast::Call { fn_name, args }),
                span: _,
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
                kind: AstKind::Submodel(ast::Submodel {
                    name,
                    local_name,
                    args,
                }),
                span,
            }
        }

        //rate_equation = { "dot" ~ "(" ~ name ~ ")" ~ "+=" ~ expression }
        Rule::rate_equation => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            Ast {
                kind: AstKind::RateEquation(ast::RateEquation {
                    name,
                    rhs: Box::new(parse_value(inner.next().unwrap())),
                }),
                span,
            }
        }

        //equation   = { expression ~ "=" ~ expression }
        Rule::equation => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::Equation(ast::Equation {
                    lhs: Box::new(parse_value(inner.next().unwrap())),
                    rhs: Box::new(parse_value(inner.next().unwrap())),
                }),
                span,
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
                let subspan = Some(StringSpan {
                    pos_start: head_term.span.unwrap().pos_start,
                    pos_end: rhs_term.span.unwrap().pos_end,
                });
                head_term = Ast {
                    kind: AstKind::Binop(ast::Binop {
                        op: term_op,
                        left: Box::new(head_term),
                        right: Box::new(rhs_term),
                    }),
                    span: subspan,
                };
            }
            if sign.is_some() {
                Ast {
                    kind: AstKind::Monop(ast::Monop {
                        op: sign.unwrap(),
                        child: Box::new(head_term),
                    }),
                    span,
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
                let subspan = Some(StringSpan {
                    pos_start: head_factor.span.unwrap().pos_start,
                    pos_end: rhs_factor.span.unwrap().pos_end,
                });
                head_factor = Ast {
                    kind: AstKind::Binop(ast::Binop {
                        op: factor_op,
                        left: Box::new(head_factor),
                        right: Box::new(rhs_factor),
                    }),
                    span: subspan,
                };
            }
            head_factor
        }

        // factor     = { call | name | real | integer | "(" ~ expression ~ ")" }
        Rule::factor => parse_value(pair.into_inner().next().unwrap()),

        _ => unreachable!("{:?}", pair.to_string()),
    }
}



pub fn parse_string(text: &str) -> Result<Vec<Box<Ast>>, Error<Rule>> {
    let main = MsParser::parse(Rule::main, &text)?.next().unwrap();
    let ast_nodes= main
        .into_inner()
        .take_while(|pair| pair.as_rule() != Rule::EOI)
        .map(parse_value)
        .map(Box::new)
        .collect();
    return Ok(ast_nodes);
}

#[cfg(test)]
mod tests {
    use super::parse_string;
    use crate::{ast::AstKind, ast::Model, ast::Ast};

    fn ast_to_model(node: Box<Ast>) -> Model {
        if let AstKind::Model(model) = node.kind {
            model
        } else {
            unreachable!()
        }
    }

    #[test]
    fn empty_model() {
        const TEXT: &str = "model test() {}";
        let models: Vec<Model> = parse_string(TEXT).unwrap().into_iter().map(ast_to_model).collect();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test");
        assert!(models[0].unknowns.is_empty());
        assert!(models[0].statements.is_empty());
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
        let models: Vec<Model> = parse_string(text).unwrap().into_iter().map(ast_to_model).collect();
        assert_eq!(models.len(), 2);

        assert_eq!(models[0].name, "capacitor");
        assert_eq!(models[0].unknowns.len(), 3);
        if let AstKind::Unknown(unknown) = &models[0].unknowns[0].kind {
            assert_eq!(unknown.name, "i");
            assert_eq!(unknown.dependents.len(), 1);
            assert!(unknown.codomain.is_none());
        } else {
            panic!("should be unknown");
        }
        if let AstKind::Unknown(unknown) = &models[0].unknowns[1].kind {
            assert_eq!(unknown.name, "v");
            assert_eq!(unknown.dependents.len(), 1);
            assert!(unknown.codomain.is_none());
        } else {
            panic!("should be unknown");
        }
        if let AstKind::Unknown(unknown) = &models[0].unknowns[2].kind {
            assert_eq!(unknown.name, "c");
            assert_eq!(unknown.dependents.len(), 0);
            assert!(unknown.codomain.is_some());
        } else {
            panic!("should be unknown");
        }
        assert_eq!(models[0].statements.len(), 1);
        if let AstKind::Equation(eqn) = &models[0].statements[0].kind {
            assert!(matches!(eqn.lhs.kind, AstKind::Name(name) if name == "i"));
            assert!(matches!(&eqn.rhs.kind, AstKind::Binop(binop) if binop.op == '*'));
        } else {
            assert!(false, "not an equation")
        }
    }

    #[test]
    fn rate_equation() {
        let text = "
        model diffusion( x -> Omega, d -> NonNegative, y(x) ) { 
            dot(y) = d * div(grad(y, x), x) 
        }
        ";
        let models: Vec<Model> = parse_string(text).unwrap().into_iter().map(ast_to_model).collect();
        assert_eq!(models.len(), 1);

        assert_eq!(models[0].name, "diffusion");
        assert_eq!(models[0].unknowns.len(), 3);
        assert_eq!(models[0].statements.len(), 1);
        if let AstKind::RateEquation(reqn) = &models[0].statements[0].kind {
            assert_eq!(reqn.name, "y");
            assert!(matches!(&reqn.rhs.kind, AstKind::Binop(binop) if binop.op == '*'));
        } else {
            assert!(false, "not a rate equation")
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
        let models: Vec<Model> = parse_string(text).unwrap().into_iter().map(ast_to_model).collect();
        assert_eq!(models.len(), 2);

        assert_eq!(models[1].name, "circuit");
        assert_eq!(models[1].unknowns.len(), 3);
        assert_eq!(models[1].statements.len(), 2);
        if let AstKind::Definition(dfn) = &models[1].statements[0].kind {
            assert_eq!(dfn.name, "inputVoltage");
            assert!(
                matches!(&dfn.rhs.kind, AstKind::Call(call) if call.fn_name == "sin" && call.args.len() == 1)
            );
        } else {
            assert!(false, "not an definition")
        }
        if let AstKind::Submodel(submodel) = &models[1].statements[1].kind {
            assert_eq!(submodel.name, "resistor");
            assert_eq!(submodel.local_name, "resistor");
            assert_eq!(submodel.args.len(), 1);
            if let AstKind::CallArg(arg) = &submodel.args[0].kind {
                assert_eq!(arg.name.unwrap(), "v");
                assert!(
                    matches!(arg.expression.kind, AstKind::Name(name) if name == "inputVoltage")
                );
            } else {
                unreachable!("not a call arg")
            }
        } else {
            assert!(false, "not an definition")
        }
    }
}

