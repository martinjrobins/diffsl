use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use pest::Span;
use std::boxed::Box;
use std::collections::HashMap;

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
        match &node.kind {
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
    name: &'s str,
    variables: Vec<Variable<'s, 'a>>,
    stmts: Vec<Ast<'a>>,
    output: Vec<Output<'s, 'a>>,
}

impl<'s, 'a> ModelInfo<'s, 'a> {
    pub fn build(
        name: &'s str,
        models: &Vec<&'a ast::Model<'s>>,
    ) -> Option<Self> {
        match models.iter().position(|v| v.name == name) {
            Some(i) => {
                let other_models = [&models[..i], &models[i..]].concat();
                Some(Self::builder(models[i], &other_models))
            }
            None => None,
        }
    }

    fn builder(
        model: &'a ast::Model<'s>,
        models: &Vec<&'a ast::Model<'s>>,
    ) -> Self {
        let mut info = Self {
            name: model.name,
            output: Vec::new(),
            stmts: Vec::new(),
            variables: Variable::new_from_vec(model.unknowns.as_slice()),
        };
        for stmt in model.statements.iter() {
            match &stmt.kind {
                AstKind::Submodel(submodel_call) => {
                    // find name in models
                    let mut submodel = match Self::build(submodel_call.name, models) {
                        Some(x) => x,
                        None => {
                            info.output.push(Output {
                                text: format!("Submodel name {} not found", submodel_call.name),
                                ast_node: &stmt,
                            });
                            continue;
                        }
                    };
                    info.add_submodel(&mut submodel, &submodel_call);
                }
                AstKind::Equation(eqn) => {
                    info.check_expr(&eqn.lhs);
                    info.check_expr(&eqn.rhs);
                    info.stmts.push(*stmt.clone());
                }
                AstKind::RateEquation(reqn) => {
                    // check name exists
                    info.check_expr(&reqn.rhs);
                    info.stmts.push(*stmt.clone());
                }
                AstKind::Definition(_) => {
                    info.variables.push(Variable::new(&stmt));
                    info.stmts.push(*stmt.clone());
                }
                _ => (),
            }
        }
        info
    }

    fn add_submodel<'mi>(
        &mut self,
        submodel: &'mi mut ModelInfo<'s, 'a>,
        submodel_call: &'a ast::Submodel<'s>,
    ) {
        let replacements = self.find_replacements(submodel, submodel_call);
        self.output.append(&mut submodel.output);
        for stmt in &submodel.stmts {
            self.stmts.push(stmt.clone_and_subst(&replacements));
        }
    }

    fn check_expr(& mut self, expr: &'a Box<Ast<'s>>) {
        match &expr.kind {
            AstKind::Name(name) => {
                // check name exists
                if self.variables.iter().find(|v| v.name == *name).is_none() {
                    self.output.push(Output {
                        text: format!("name {} not found", name),
                        ast_node: expr,
                    })
                }
            }
            AstKind::Binop(binop) => {
                self.check_expr(&binop.left);
                self.check_expr(&binop.right);
            }
            AstKind::Monop(monop) => {
                self.check_expr(&monop.child);
            }
            _ => (),
        }
    }
    fn find_replacements<'mi>(
        & mut self,
        submodel: &'mi ModelInfo<'s, 'a>,
        submodel_call: &'a ast::Submodel<'s>,
    ) -> HashMap<&'s str, &'a Box<Ast<'s>>> {
        let mut replacements = HashMap::new();
        let mut found_kwarg = false;

        // find all the replacements specified in the call arguments
        for (i, arg) in submodel_call.args.iter().enumerate() {
            if let AstKind::CallArg(call_arg) = &arg.kind {
                if let Some(name) = call_arg.name {
                    found_kwarg = true;
                    if let Some(_) = submodel.variables.iter().find(|v| v.name == name) {
                        replacements.insert(name, &call_arg.expression);
                    } else {
                        self.output.push(Output {
                            text: format!(
                                "Cannot find unknown {} in model {}",
                                name, submodel.name
                            ),
                            ast_node: arg,
                        });
                    }
                } else {
                    if found_kwarg {
                        self.output.push(Output {
                            text: format!("positional argument found after keyword argument"),
                            ast_node: arg,
                        });
                    }
                    replacements.insert(submodel.variables[i].name, &call_arg.expression);
                };
            }
        }
        replacements
    }
}





#[cfg(test)]
mod tests {
    use crate::{builder::ModelInfo, parser::parse_string, ast::Model, ast::AstKind};

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
        let models_ref: Vec<&Model> = models.iter().collect();
        let model_info = ModelInfo::build("circuit", &models_ref).unwrap();
        assert_eq!(model_info.variables.len(), 4);
        assert_eq!(model_info.stmts.len(), 1);
        assert_eq!(model_info.output.len(), 1);
        assert!(model_info.output[0].text.contains("resistorr") == true);
    }
    #[test]
    fn submodel_replacements() {
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
        let models_ref: Vec<&Model> = models.iter().collect();
        let model_info = ModelInfo::build("circuit", &models_ref).unwrap();
        assert_eq!(model_info.variables.len(), 4);
        assert_eq!(model_info.stmts.len(), 2);
        if let AstKind::Equation(eqn) = &model_info.stmts[1].kind {
            assert!(
                matches!(&eqn.lhs.kind, AstKind::Name(name) if *name == "inputVoltage")
            );
            assert!(
                matches!(&eqn.rhs.kind, AstKind::Binop(binop) if binop.op == '*')
            );
        } else {
            assert!(false, "not an equation")
        }
        assert_eq!(model_info.output.len(), 0);
    }
    #[test]
    fn variable_name_not_found() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * doesnotexist
        }
        ";
        let models = parse_string(text).unwrap();
        let models_ref: Vec<&Model> = models.iter().collect();
        let model_info = ModelInfo::build("resistor", &models_ref).unwrap();
        assert_eq!(model_info.variables.len(), 3);
        assert_eq!(model_info.output.len(), 1);
        assert!(model_info.output[0].text.contains("doesnotexist") == true);
    }
}
