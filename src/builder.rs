use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast;
use std::boxed::Box;
use std::intrinsics::unreachable;
use std::collections::HashMap;
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
            AstKind::Unknown(unknown) => Variable {
                name: unknown.name,
                ast_node: node,
            },
            AstKind::Definition(dfn) => Variable {
                name: dfn.name,
                ast_node: node,
            },
            _ => panic!("Cannot create variable from {}", node),
        }
    }
}

#[derive(Debug)]
pub struct ModelInfo<'s, 'a> {
    variables: Vec<Variable<'s, 'a>>,
    eqns: Vec<Ast<'a>>,
    output: Vec<Output<'s, 'a>>,
}

fn info_from_model<'s, 'a>(model: &'a ast::Model<'s>) -> ModelInfo<'s, 'a> {
    ModelInfo {
        output: Vec::new(),
        eqns: Vec::new(),
        variables: Variable::new_from_vec(model.unknowns.as_slice()),
    }
}

pub fn builder_index<'s, 'a>(index: int, models: &'a Vec<ast::Model<'s>>) -> Vec<ModelInfo<'s, 'a>> {
    let mut ret: Vec<ModelInfo> = models.iter().map(info_from_model).collect();
    // split vector of models into [left, this_model, right]
    // then combine left and right into other_models = [left, right]
    let (left, right_inclusive) = models.split_at(index);
    if let Some((this_model, right)) = right_inclusive.split_first() {
        let other_models = left.iter().chain(right.iter());
        let info = &mut ret[i];
        builder(this_model, other_models, info);
    } else {
        unreachable!()
    }
    ret
}

fn examine_expression<'s, 'a, 'mi>(expr: &'a Box<Ast<'s>>, info: &'mi mut ModelInfo<'s, 'a>) {
    match &expr.kind {
        AstKind::Name(name) => {
            // check name exists
            if info.variables.iter().find(|v| v.name == *name).is_none() {
                info.output.push(Output {
                    text: format!("name {} not found", name),
                    ast_node: expr,
                })
            }
        }
        AstKind::Binop(binop) => {
            examine_expression(&binop.left, info);
            examine_expression(&binop.right, info);
        }
        AstKind::Monop(monop) => {
            examine_expression(&monop.child, info);
        }
        _ => (),
    }
}



fn builder<'s, 'a, 'mi, I>(model: &'a ast::Model<'s>, models: I, info: &'mi mut ModelInfo<'s, 'a>)
where
    I: Iterator<Item = &'a ast::Model<'s>> + Clone,
{
    for stmt in model.statements {
        match stmt.kind {
            AstKind::Submodel(submodel) => {
                // find name in models
                let model_match = models.find(|m| m.name == submodel.name);
                if model_match.is_none() {
                    info.output.push(Output {
                        text: format!("Submodel name {} not found", submodel.name),
                        ast_node: &stmt,
                    })
                } else {
                    add_submodel_equations(model_match.unwrap(), &submodel, info);
                }
                // check all unknowns are in call args
            }
            AstKind::Equation(eqn) => {
                examine_expression(&eqn.lhs, info);
                examine_expression(&eqn.rhs, info);
                info.eqns.push(stmt.clone());
            }
            AstKind::RateEquation(reqn) => {
                // check name exists
                examine_expression(&reqn.rhs, info);
                info.eqns.push(stmt.clone());
            }
            AstKind::Definition(dfn) => {
                info.variables.push(Variable::new(stmt));
                info.eqns.push(stmt.clone());
            },
            _ => (),
        }
    }
}


fn add_submodel_equations<'s, 'a, 'mi>(model: &'a ast::Model<'s>,  model_call: &'a ast::Submodel<'s>, info: &'mi mut ModelInfo<'s, 'a>) {
    let mut found_kwarg = false;
    let mut replacements = HashMap::new();

    // find all the replacements specified in the call arguments
    for (i, arg) in model_call.args.iter().enumerate() {
        if let AstKind::CallArg(call_arg) = arg.kind {
            if let Some(name) = call_arg.name {
                found_kwarg = true;
                let find_unknown = model.unknowns.into_iter().map(|u| {
                    match u.kind { 
                        AstKind::Unknown(unknown) => unknown,
                        _ => unreachable!()
                    }}).find(|u| u.name == name);
                if let Some(unknown) = find_unknown {
                    replacements.insert(name, call_arg.expression);
                } else {
                    info.output.push(Output {
                        text: format!("Cannot find unknown {} in model {}", name, model.name),
                        ast_node: arg,
                    });
                }

            } else {
                if found_kwarg {
                    info.output.push(Output {
                        text: format!("positional argument found after keyword argument"),
                        ast_node: arg,
                    })
                }
                let unknown = if let AstKind::Unknown(unknown) = model.unknowns[i].kind {
                    unknown
                } else {
                    unreachable!()
                };
                replacements.insert(unknown.name, call_arg.expression);
            };
        }
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
    #[test]
    fn variable_name_not_found() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * doesnotexist
        }
        ";
        let models = parse_string(text).unwrap();
        let model_infos = semantic_pass(&models);
        assert_eq!(model_infos.len(), 1);
        assert_eq!(model_infos[0].variables.len(), 3);
        assert_eq!(model_infos[0].output.len(), 1);
        assert!(model_infos[0].output[0].text.contains("doesnotexist") == true);
    }
}
