use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Call;
use crate::ast::Model;
use pest::Span;
use std::boxed::Box;
use std::cmp::max;
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

#[derive(Debug, Clone)]
pub struct Dependent<'s> {
    pub name: &'s str,
    pub max_derivative: usize,
    pub num_bcs: usize,
}

#[derive(Debug, Clone)]
pub struct Variable<'s, 'a> 
    where 'a: 's
{
    pub name: &'s str,
    pub dim: usize,
    pub state: bool,
    pub bounds: (f64, f64),
    pub constant: bool,
    pub dependents: Vec<Dependent<'s>>,
    pub ast_node: &'a Box<Ast<'s>>,
}

impl<'s, 'a> Variable<'s, 'a> {
    pub fn new(node: &'a Box<Ast<'s>>, info: &mut ModelInfo<'s, 's>) -> Variable<'s, 'a> {
        match &node.kind {
            AstKind::Unknown(unknown) => {
                let is_time = unknown.name == "t";
                let time_dependent = unknown.dependents.iter().any(|d| *d == "t");
                let dependents = unknown
                    .dependents
                    .iter()
                    .map(|d| Dependent {
                        name: d,
                        max_derivative: 0,
                        num_bcs: 0,
                    })
                    .collect();
                let bounds = match &unknown.codomain {
                    Some(r) => match &r.kind {
                        AstKind::Range(r) => (r.lower, r.upper),
                        AstKind::Name(name) => {
                            match *name {
                                "NonNegative" => (0.0, f64::INFINITY),
                                "R" => (-f64::INFINITY, f64::INFINITY),
                                _ => {
                                    info.output.push(Output {
                                        text: format!("Unknown domain {}", name),
                                        ast_node: node,
                                    });
                                    (-f64::INFINITY, f64::INFINITY)
                                }
                            }

                        },
                        _ => unreachable!(),
                    },
                    None => (if is_time { 0.0 } else { -f64::INFINITY }, f64::INFINITY),
                };
                Variable {
                    name: unknown.name,
                    ast_node: node,
                    dim: 1,
                    state: time_dependent,
                    constant: !time_dependent,
                    dependents,
                    bounds,
                }
            }
            AstKind::Definition(dfn) => {
                let deps = dfn.rhs.get_dependents();
                let time_dependent = deps.contains("t");
                let dependents = deps
                    .into_iter()
                    .map(|d| Dependent {
                        name: d,
                        max_derivative: 0,
                        num_bcs: 0,
                    })
                    .collect();
                let bounds = (-f64::INFINITY, f64::INFINITY);
                Variable {
                    name: dfn.name,
                    ast_node: node,
                    dim: 1,
                    state: false,
                    constant: !time_dependent,
                    dependents,
                    bounds,
                }
            }
            _ => panic!("Cannot create variable from {}", node),
        }
    }
}

#[derive(Debug)]
pub struct ModelInfo<'s, 'a>
where
    'a: 's,
{
    pub name: &'s str,
    pub variables: Vec<Variable<'s, 'a>>,
    pub stmts: Vec<Ast<'a>>,
    pub output: Vec<Output<'s, 'a>>,
    pub ast_node: &'a Box<Ast<'s>>,
}

impl<'s, 'a> ModelInfo<'s, 'a> {
    pub fn build(name: &'s str, ast: &'a Vec<Box<Ast<'s>>>) -> Result<Self, String> {
        let model_refs: Vec<&Model> = ast.iter().filter_map(|n| AstKind::model(&n.kind)).collect();
        let ast_refs: Vec<&Box<Ast>> = ast.iter().collect();
        match model_refs.iter().position(|v| v.name == name) {
            Some(i) => {
                let other_models = [&model_refs[..i], &model_refs[i..]].concat();
                let other_asts = [&ast_refs[..i], &ast_refs[i..]].concat();
                let mut model_info =
                    Self::builder(&ast[i], model_refs[i], &other_models, &other_asts);
                model_info.check_model();
                Ok(model_info)
            }
            None => Err(format!("Model name {} not found", name)),
        }
    }
    fn build_submodel(
        name: &'s str,
        models: &Vec<&'a ast::Model<'s>>,
        asts: &Vec<&'a Box<Ast<'s>>>,
    ) -> Option<Self> {
        match models.iter().position(|v| v.name == name) {
            Some(i) => {
                let other_models = [&models[..i], &models[i..]].concat();
                let other_asts = [&asts[..i], &asts[i..]].concat();
                Some(Self::builder(
                    asts[i],
                    models[i],
                    &other_models,
                    &other_asts,
                ))
            }
            None => None,
        }
    }

    fn builder(
        ast: &'a Box<Ast<'s>>,
        model: &'a ast::Model<'s>,
        models: &Vec<&'a ast::Model<'s>>,
        asts: &Vec<&'a Box<Ast<'s>>>,
    ) -> Self {
        let mut info = Self {
            name: model.name,
            output: Vec::new(),
            stmts: Vec::new(),
            variables: Vec::with_capacity(model.unknowns.len()),
            ast_node: ast,
        };

        let reserved = ["u", "dudt", "t", "F", "G", "input"];
        // create variables from unknowns
        let mut have_time_variable = false;
        for node in model.unknowns.iter() {
            // check its not in list of reserved names
            let var = Variable::new(node, &mut info);
            if reserved.contains(&var.name) {
                // time is ok to allow to override initial value
                if var.name == "t" {
                    have_time_variable = true;
                } else {
                    info.output.push(Output {
                        text: format!("Name {} is reserved", var.name),
                        ast_node: var.ast_node,
                    });
                }
            }
            info.variables.push(var);
        }
        if !have_time_variable {
            info.variables.push(Variable {
                name: "t",
                dim: 1,
                state: false,
                bounds: (0.0, f64::INFINITY),
                constant: false,
                dependents: Vec::new(),
                ast_node: info.ast_node,
            });
        }

        for stmt in model.statements.iter() {
            match &stmt.kind {
                AstKind::Submodel(submodel_call) => {
                    // find name in models
                    let mut submodel = match Self::build_submodel(submodel_call.name, models, asts)
                    {
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
                    // its an initial condition if:
                    //  - the lhs is a call with a name equal to one of the variables,
                    //  - that variable has a dependent t,
                    //  - there is a number equal to the lower bound of time in the argument corresponding to time
                    let mut is_ic = false;

                    if let AstKind::Call(Call { fn_name, args }) = &eqn.lhs.kind {
                        if let Some(v) = info.variables.iter().find(|v| v.name == *fn_name) {
                            if let Some(t_index) = v.dependents.iter().position(|d| d.name == "t") {
                                let time = match info.variables.iter().find(|v| v.name == "t") {
                                    Some(t) => t,
                                    None => unreachable!(),
                                };
                                if let AstKind::Number(value) = args[t_index].kind {
                                    if value == time.bounds.0 {
                                        is_ic = true;
                                    }
                                }
                            }
                        }
                    }
                    if !is_ic {
                        info.check_expr(&eqn.lhs);
                    }
                    info.check_expr(&eqn.rhs);

                    info.stmts.push(*stmt.clone());
                }
                AstKind::RateEquation(reqn) => {
                    // check name exists and variable depends on time
                    match info.variables.iter_mut().find(|v| v.name == reqn.name) {
                        Some(v) => match v.dependents.iter_mut().find(|d| d.name == "t") {
                            Some(d) => d.max_derivative = max(d.max_derivative, 1),
                            None => info.output.push(Output {
                                text: format!(
                                    "Rate equation invalid: variable {} does not depend on time",
                                    v.name
                                ),
                                ast_node: stmt,
                            }),
                        },
                        None => info.output.push(Output {
                            text: format!("name {} not found", reqn.name),
                            ast_node: stmt,
                        }),
                    }
                    info.check_expr(&reqn.rhs);
                    info.stmts.push(*stmt.clone());
                }
                AstKind::Definition(_) => {
                    let var = Variable::new(&stmt, &mut info);
                    if reserved.contains(&var.name) {
                        info.output.push(Output {
                            text: format!("Name {} is reserved", var.name),
                            ast_node: var.ast_node,
                        });
                    }
                    info.variables.push(var);
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

    fn check_model(&mut self) {
        // check number of equations and unknowns
        let n_eqns = self
            .stmts
            .iter()
            .filter(|s| matches!(s.kind, AstKind::Equation(_)))
            .count();
        let n_unknowns = self.variables.iter().filter(|v| v.state).count();
        if n_eqns < n_unknowns {
            self.output.push(Output {
                text: format!(
                    "Model is underdetermined, only {} equations for {} unknowns",
                    n_eqns, n_unknowns
                ),
                ast_node: self.ast_node,
            });
        } else if n_eqns > n_unknowns {
            self.output.push(Output {
                text: format!(
                    "Model is overdetermined, only {} equations for {} unknowns",
                    n_eqns, n_unknowns
                ),
                ast_node: self.ast_node,
            });
        }
        // check that variables with derivatives have the right number of boundary conditions
        todo!()
    }

    fn check_expr(&mut self, expr: &'a Box<Ast<'s>>) {
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
            AstKind::Call(call) => {
                // check name in allowed functions
                let functions = ["sin"];
                if !functions.contains(&call.fn_name) {
                    if let Some(_) = self.variables.iter().find(|v| v.name == call.fn_name) {
                        self.output.push(Output {
                            text: format!("Invalid use of variable {}, please use \"{}\" by itself without referring to dependent variables", call.fn_name, call.fn_name),
                            ast_node: expr,
                        })
                    } else {
                        self.output.push(Output {
                            text: format!("Function or variable {} not found", call.fn_name),
                            ast_node: expr,
                        })
                    }
                }
                for arg in &call.args {
                    self.check_expr(arg);
                }
            }
            _ => unreachable!(),
        }
    }
    fn find_replacements<'mi>(
        &mut self,
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
    use crate::{ast::AstKind, builder::ModelInfo, parser::parse_string};

    #[test]
    fn simple_circuit() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i(t)) {
            let inputVoltage = sin(t) 
            use resistor(v = inputVoltage)
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.variables.len(), 2);
        assert_eq!(model_info.stmts.len(), 2);
        assert_eq!(model_info.output.len(), 0);
    }
    #[test]
    fn missing_initial_condition() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t) ) { 
            dot(y) = r * y * (1 - y / k)
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.variables.len(), 3);
        assert_eq!(model_info.stmts.len(), 1);
        for o in model_info.output.iter() {
            println!("{}", o.as_error_message(text));
        }
        assert_eq!(model_info.output.len(), 1);
    }
    #[test]
    fn submodel_name_not_found() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i(t)) {
            let inputVoltage = sin(t) 
            use resistorr(v = inputVoltage)
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.variables.len(), 2);
        assert_eq!(model_info.stmts.len(), 1);
        assert_eq!(model_info.output.len(), 2);
        assert!(model_info.output[0].text.contains("resistorr") == true);
        assert!(model_info.output[1].text.contains("underdetermined") == true);
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
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.variables.len(), 4);
        assert_eq!(model_info.stmts.len(), 2);
        if let AstKind::Equation(eqn) = &model_info.stmts[1].kind {
            assert!(matches!(&eqn.lhs.kind, AstKind::Name(name) if *name == "inputVoltage"));
            assert!(matches!(&eqn.rhs.kind, AstKind::Binop(binop) if binop.op == '*'));
        } else {
            assert!(false, "not an equation")
        }
        assert_eq!(model_info.output.len(), 1);
        assert!(
            model_info.output[0]
                .text
                .contains("Model is underdetermined")
                == true
        );
    }
    #[test]
    fn variable_name_not_found() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * doesnotexist
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("resistor", &models).unwrap();
        assert_eq!(model_info.variables.len(), 3);
        assert_eq!(model_info.output.len(), 2);
        assert!(model_info.output[0].text.contains("doesnotexist") == true);
        assert!(model_info.output[1].text.contains("underdetermined") == true);
    }
}
