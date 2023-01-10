use crate::pest::Parser;

#[derive(Parser)]
#[grammar = "ds_grammar.pest"] // relative to src
pub(crate) struct DsParser;

use crate::ast::StringSpan;

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
        Rule::name => Ast {
            kind: AstKind::Name(pair.as_str()),
            span,
        },

        // integer    = @{ ('0'..'9')+ }
        // real       = @{ ( ('0'..'9')+ ~ "." ~ ('0'..'9')+ ) | integer }
        Rule::integer | Rule::real => Ast {
            kind: AstKind::Number(pair.as_str().parse().unwrap()),
            span,
        },

        // domain = { "[" ~ real ~ "..." ~ real ~ "]" }
        Rule::domain => {
            let mut inner = pair.into_inner();
            Ast {
                kind: AstKind::Range(ast::Range {
                    lower: inner.next().unwrap().as_str().parse().unwrap(),
                    upper: inner.next().unwrap().as_str().parse().unwrap(),
                }),
                span,
            }
        }

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

        // array      = { name ~ "{" ~ array_elmt? ~ (DELIM~ array_elmt )* ~ DELIM? ~ "}" }
        Rule::array =>  {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str();
            let elmts = inner.map(|v| Box::new(parse_value(v))).collect();
            Ast { 
                kind: AstKind::Array(ast::Array { name, elmts, }),
                span 
            }
        },

        // array_elmt = { expression | parameter | assignment }
        Rule::array_elmt =>  {
            parse_value(pair.into_inner().next().unwrap())
        },

        //parameter  = { name ~ "->" ~  domain }
        Rule::parameter => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str();
            let range = Box::new(parse_value(inner.next().unwrap()));
            Ast { 
                kind: AstKind::Parameter(ast::Parameter { name, range }),
                span 
            }

        },

        //assignment = { name ~ "=" ~ expression }
        Rule::assignment => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str();
            let expr = Box::new(parse_value(inner.next().unwrap()));
            Ast { 
                kind: AstKind::Assignment(ast::Assignment{ name, expr }),
                span 
            }
        },

        _ => unreachable!("{:?}", pair.to_string()),
    }
}



pub fn parse_string(text: &str) -> Result<Vec<Box<Ast>>, Error<Rule>> {
    let main = DsParser::parse(Rule::main, &text)?.next().unwrap();
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
    use crate::{ast::{Array}};

    #[test]
    fn basic_model() {
        const TEXT: &str = "
            in {}
            test {
                1,
            }
        ";
        let arrays: Vec<Array> = parse_string(TEXT).unwrap().into_iter().map(|a| a.kind.into_array().unwrap()).collect();
        println!("{:?}", arrays);
    }

    
}
