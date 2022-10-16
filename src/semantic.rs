use crate::ast::Ast;
use crate::ast::AstKind;
use std::boxed::Box;
use pest::Span;

#[derive(Debug)]
pub struct Output<'s, 'a> {
    text: String,
    ast_node: &'a Box<Ast<'s>>,
}

impl<'s, 'a> Output<'s, 'a> {
    pub fn as_error_message(self: &Output<'s, 'a>, input: &'a str) -> String {
        let string_span = &self.ast_node.as_ref().span;
        let span = Span::new(input, string_span.pos_start, string_span.pos_end);
        let (line, col) = span.as_ref().unwrap().start_pos().line_col();
        format!("Line {}, Column {}: Error: {}", line, col, self.text)
    }
}
 

#[derive(Debug)]
pub struct Variable<'s, 'a> {
    name: &'s str,
    ast_node: &'a Box<Ast<'s>>,
}

impl<'s, 'a> Variable<'s, 'a> {
    pub fn new_from_vec(nodes: &'a [Box<Ast<'s>>]) -> Vec<Variable<'s, 'a>> {
        let mut ret: Vec<Variable> = Vec::with_capacity(nodes.len());
        for node in nodes {
            ret.push(Variable::new(node));
        }
        ret
    }
    pub fn new(node: &'a Box<Ast<'s>>) -> Variable<'s, 'a> {
        match node.kind {
            AstKind::Unknown {
                name,
                dependents: _,
                codomain: _,
            } => Variable {
                name,
                ast_node: node,
            },
            AstKind::Definition { name, rhs: _ } => Variable {
                name,
                ast_node: node,
            },
            _ => panic!("Cannot create variable from {}", node),
        }
    }
}

#[derive(Debug)]
pub struct ModelInfo<'s, 'a> {
    variables: Vec<Variable<'s, 'a>>,
    output: Vec<Output<'s, 'a>>,
}

fn info_from_model<'s, 'a>(model: &'a Ast<'s>) -> ModelInfo<'s, 'a> {
    if let AstKind::Model {
        name: _,
        unknowns,
        statements: _,
    } = &model.kind
    {
        ModelInfo {
            output: Vec::new(),
            variables: Variable::new_from_vec(unknowns.as_slice()),
        }
    } else {
        unreachable!()
    }
}

pub fn semantic_pass<'s, 'a>(models: &'a Vec<Ast<'s>>) -> Vec<ModelInfo<'s, 'a>> {
    let mut ret: Vec<ModelInfo> = models.iter().map(info_from_model).collect();
    for i in 0..models.len() {
        // split vector of models into [left, this_model, right]
        // then combine left and right into other_models = [left, right]
        let (left, right_inclusive) = models.split_at(i);
        if let Some((this_model, right)) = right_inclusive.split_first() {
            let other_models = left.iter().chain(right.iter());
            let info = &mut ret[i];
            examine_model(this_model, other_models, info);
        } else {
            unreachable!()
        }
    }
    ret
}

fn examine_expression<'s, 'a, 'mi>(expr: &'a Box<Ast<'s>>, info: &'mi mut ModelInfo<'s, 'a>) {
    match &expr.kind {
        // check name exists
        AstKind::Name(name) => {
            if info.variables.iter().find(|v| v.name == *name).is_none() {
                info.output.push(Output {
                    text: format!("name {} not found", name),
                    ast_node: expr,
                })
            }
        }
        AstKind::Binop { op: _, left, right } => {
            examine_expression(&left, info);
            examine_expression(&right, info);
        }
        AstKind::Monop { op: _, child } => {
            examine_expression(&child, info);
        }
        _ => (),
    }
}

fn examine_model<'s, 'a, 'mi, I>(model: &'a Ast<'s>, models: I, info: &'mi mut ModelInfo<'s, 'a>)
where
    I: Iterator<Item = &'a Ast<'s>> + Clone,
{
    if let AstKind::Model {
        name: _,
        unknowns: _,
        statements,
    } = &model.kind
    {
        for stmt in statements {
            match &stmt.kind {
                AstKind::Submodel {
                    name,
                    local_name: _,
                    args: _,
                } => {
                    // check that name in models
                    if models
                        .clone()
                        .find(|m| match &m.kind {
                            AstKind::Model {
                                name: this_name,
                                unknowns: _,
                                statements: _,
                            } => this_name == name,
                            _ => false,
                        })
                        .is_none()
                    {
                        info.output.push(Output {
                            text: format!("Submodel name {} not found", name),
                            ast_node: stmt,
                        })
                    }
                    // check all unknowns are in call args
                }
                AstKind::Equation { lhs, rhs } => {
                    examine_expression(&lhs, info);
                    examine_expression(&rhs, info);
                }
                AstKind::RateEquation { name: _, rhs } => {
                    // check name exists
                    examine_expression(&rhs, info);
                }
                AstKind::Definition { name: _, rhs: _ } => info.variables.push(Variable::new(stmt)),
                _ => (),
            }
        }
    } else {
        unreachable!();
    }
}

#[cfg(test)]
mod tests {
    use crate::{parser::parse_string, semantic::semantic_pass};

    #[test]
    fn submodel_name_not_found() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i1(t), i2(t), i3(t)) {
            let inputVoltage = sin(t) 
            use resistorr(v = inputVoltage)
        }
        ";
        let models = parse_string(text).unwrap();
        let model_infos = semantic_pass(&models);
        assert_eq!(model_infos.len(), 2);
        assert_eq!(model_infos[0].variables.len(), 3);
        assert_eq!(model_infos[0].output.len(), 0);
        assert_eq!(model_infos[1].variables.len(), 4);
        assert_eq!(model_infos[1].output.len(), 1);
        assert!(model_infos[1].output[0].text.contains("resistorr") == true);
    }
}
