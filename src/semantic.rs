use crate::ast::Ast;
use crate::ast::AstKind;
use std::rc::Rc;

#[derive(Debug)]
pub struct Output<'a> {
    text: String,
    reference: Rc<Ast<'a>>,
}

#[derive(Debug)]
pub struct Variable<'a> {
    name: &'a str,
    reference: Rc<Ast<'a>>,
}

impl<'a> Variable<'a> {
    pub fn new(node: Rc<Ast<'a>>) -> Variable<'a> {
        match node.kind {
            AstKind::Unknown {
                name,
                dependents,
                codomain,
            } => Variable {
                name,
                reference: node,
            },
            AstKind::Definition { name, rhs } => Variable {
                name,
                reference: node,
            },
            _ => panic!("Cannot create variable from {}", node),
        }
    }
}

#[derive(Debug)]
pub struct ModelInfo<'a> {
    variables: Vec<Variable<'a>>,
    output: Vec<Output<'a>>,
}

pub fn semantic_pass(models: &mut Vec<Ast>) {
    for model in models {
        examine_model(model, models);
    }
}

fn examine_expression(expr: Ast, info: &mut ModelInfo) {
    match expr.kind {
        // check name exists
        AstKind::Name(name) => {
            if info.variables.iter().find(|v| v.name == name).is_none() {
                info.output.push(Output {
                    text: format!("name {} not found", name),
                    reference: expr,
                })
            }
        }
        AstKind::Binop { op, left, right } => {
            examine_expression(left, info);
            examine_expression(right, info);
        }
        AstKind::Monop { op, child } => {
            examine_expression(child, info);
        }
        _ => (),
    }
}

fn examine_model(model: &mut Ast, models: &Vec<Ast>) {
    let mut info = if let AstKind::Model {
        name,
        unknowns,
        statements,
        info,
    } = model.kind
    {
        info.insert(ModelInfo {
            output: Vec::new(),
            variables: unknowns.map(Variable::new),
        });
        for stmt in statements.as_ref() {
            match stmt.kind {
                AstKind::Submodel {
                    name,
                    local_name,
                    args,
                } => {
                    // check that name in models
                    if models
                        .iter()
                        .find(|&&m| match m.kind {
                            AstKind::Model {
                                name: this_name,
                                unknowns,
                                statements,
                                info,
                            } => this_name == *name,
                            _ => false,
                        })
                        .is_none()
                    {
                        info.unwrap().output.push(Output {
                            text: format!("Submodel name {} not found", *name),
                            reference: stmt,
                        })
                    }
                    // check all unknowns are in call args
                }
                AstKind::Equation { lhs, rhs } => {
                    examine_expression(lhs, &mut info.unwrap());
                    examine_expression(rhs, &mut info.unwrap());
                }
                AstKind::RateEquation { name, rhs } => {
                    // check name exists
                    examine_expression(rhs, &mut info.unwrap());
                }
                AstKind::Definition { name, rhs } => {
                    info.unwrap().variables.push(Variable::new(stmt))
                }
                _ => (),
            }
        }
    } else {
        unreachable!();
    };
}
