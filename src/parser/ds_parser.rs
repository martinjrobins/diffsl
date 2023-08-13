#[derive(Parser)]
#[grammar = "ds_grammar.pest"] // relative to src
pub struct DsParser;

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


        // name_ij    = { name ~ ("_" ~ name)? }
        Rule::name_ij => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let indices = if inner.peek().is_some() {
                let indices = parse_name(inner.next().unwrap());
                indices.chars().collect::<Vec<_>>()
            } else {
                vec![]
            };
            Ast {
                kind: AstKind::IndexedName(ast::IndexedName { name, indices }),
                span,
            }
        }

        
        //model      = { inputs? ~ tensor* }
        //inputs     = { "inputs" ~ "=" ~ "[" ~ name ~ ("," ~ name)* ~ "]" }
        // TODO: refactor inputs to an ast node
        Rule::model => {
            let mut inner = pair.into_inner();
            let inputs = if inner.peek().is_some() {
                if inner.peek().unwrap().as_rule() == Rule::inputs {
                    inner.next().unwrap().into_inner().map(parse_name).collect()
                } else {
                    vec![]
                }
            } else {
                vec![]
            };
            let tensors: Vec<Box<Ast>> = inner.map(parse_value).map(Box::new).collect();
            Ast {
                kind: AstKind::DsModel(ast::DsModel { inputs, tensors }),
                span,
            }
        }

        // tensor     = { name_ij ~ "{" ~ tensor_elmt? ~ (DELIM~ tensor_elmt )* ~ DELIM? ~ "}" }
        Rule::tensor =>  {
            let mut inner = pair.into_inner();
            let name_ij = parse_value(inner.next().unwrap());
            let (name, indices) = match name_ij.kind {
                AstKind::IndexedName(ast::IndexedName { name, indices }) => {
                    (name, indices)
                }
                _ => unreachable!()
            };
            let elmts = inner.map(|v| Box::new(parse_value(v))).collect();
            Ast { 
                kind: AstKind::Tensor(ast::Tensor { name, elmts, indices }),
                span 
            }
        },

        // indice      = { integer ~ ( range_sep ~ integer )? }
        Rule::indice => {
            let mut inner = pair.into_inner();
            let first = Box::new(parse_value(inner.next().unwrap()));
            if inner.peek().is_some() {
                let sep = Some(inner.next().unwrap().as_str());
                let last = Some(Box::new(parse_value(inner.next().unwrap())));
                Ast { 
                    kind: AstKind::Indice(ast::Indice { first, last, sep }),
                    span
                }
            } else {
                Ast { 
                    kind: AstKind::Indice(ast::Indice { first, last: None, sep: None }),
                    span 
                }
            }
        },

        // indices   = { "(" ~ indice ~ ("," ~ indice)* ~ ")" ~ ":" }
        Rule::indices => {
            let inner = pair.into_inner();
            let indices = inner.map(|v| Box::new(parse_value(v))).collect::<Vec<_>>();
            Ast { 
                kind: AstKind::Vector(ast::Vector { data: indices }),
                span 
            }
        },

        // assignment = { name_ij ~ "=" ~ expression }
        Rule::assignment => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let expr = Box::new(parse_value(inner.next().unwrap()));
            Ast { 
                kind: AstKind::Assignment(ast::Assignment { name, expr }),
                span 
            }
        },

        // tensor_elmt = { indices? ~ (assignment | expression) }
        Rule::tensor_elmt =>  {
            let mut inner = pair.into_inner();
            let indices = if inner.peek().unwrap().as_rule() == Rule::indices {
                Some(Box::new(parse_value(inner.next().unwrap())))
            } else {
                None
            };
            let expr = Box::new(parse_value(inner.next().unwrap()));
            Ast { 
                kind: AstKind::TensorElmt(ast::TensorElmt { indices, expr }),
                span 
            }
        },

        _ => unreachable!("{:?}", pair.to_string()),
    }
}



pub fn parse_string(text: &str) -> Result<ast::DsModel, Error<Rule>> {
    let main = DsParser::parse(Rule::main, &text)?.next().unwrap();
    let model = parse_value(main.into_inner().next().unwrap()).kind.to_ds_model().unwrap();
    Ok(model)
}

#[cfg(test)]
mod tests {

    use super::parse_string;

    #[test]
    fn basic_model() {
        const TEXT: &str = "
            test {
                1,
            }
        ";
        let model = parse_string(TEXT).unwrap();
        assert_eq!(model.inputs.len(), 0);
        assert_eq!(model.tensors.len(), 1);
        let tensor = model.tensors[0].kind.as_tensor().unwrap();
        assert_eq!(tensor.name, "test");
        assert_eq!(tensor.elmts.len(), 1);
        let tensor_elmt = tensor.elmts[0].kind.as_tensor_elmt().unwrap();
        assert!(tensor_elmt.indices.is_none());
        assert_eq!(tensor_elmt.expr.kind.as_real().unwrap(), 1.0);
    }

    #[test]
    fn logistic_model() {
        const TEXT: &str = "
            in = [r, k] 
            r { 1 }
            k { 1 }
            I_ij {
                (0, 0): 1,
                (1..2, 1..2): 1,
                (2:3, 2:3): 1,
            }
            u_i {
                y = 1,
                (1:3): z = 1,
            }
            F {
                dydt,
                0,
            }
            G {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out {
                y,
                t,
                z
            }
        ";
        let model = parse_string(TEXT).unwrap();
        assert_eq!(model.tensors.len(), 7);
        assert_eq!(model.inputs.len(), 2);
        let tensor = model.tensors[0].kind.as_tensor().unwrap();
        assert_eq!(tensor.name, "r");
        assert_eq!(tensor.elmts.len(), 1);
        let tensor = model.tensors[2].kind.as_tensor().unwrap();
        assert_eq!(tensor.name, "I");
        assert_eq!(tensor.elmts.len(), 3);
        assert_eq!(tensor.elmts[0].kind.as_tensor_elmt().unwrap().indices.as_ref().unwrap().kind.as_vector().unwrap().data.len(), 2);
        assert_eq!(tensor.elmts[0].kind.as_tensor_elmt().unwrap().expr.to_string(), "1");
        assert_eq!(tensor.elmts[1].kind.as_tensor_elmt().unwrap().expr.to_string(), "1");
        let tensor = model.tensors[3].kind.as_tensor().unwrap();
        assert_eq!(tensor.name, "u");
        assert_eq!(tensor.elmts.len(), 2);
        assert_eq!(tensor.elmts[0].kind.as_tensor_elmt().unwrap().expr.kind.as_assignment().unwrap().name, "y");
        assert_eq!(tensor.elmts[0].kind.as_tensor_elmt().unwrap().expr.kind.as_assignment().unwrap().expr.to_string(), "1");
    }


    
}
