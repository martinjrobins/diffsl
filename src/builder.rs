use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Call;
use crate::ast::Model;
use crate::ast::StringSpan;
use crate::ast::Unknown;
use pest::Span;
use std::boxed::Box;
use std::collections::HashMap;
use std::hash::Hash;
use std::ptr::null;
use std::rc::Rc;
use std::rc::Weak;

#[derive(Debug)]
pub struct Output {
    pub text: String,
    pub source_ref: StringSpan,
    pub secondary_txts: Vec<String>,
    pub secondary_refs: Vec<StringSpan>,
}

impl Output {
    pub fn new(text: String, span: StringSpan) -> Self {
        Self {
            text: text,
            source_ref: span,
            secondary_txts: Vec::new(),
            secondary_refs: Vec::new(),
        }
    }
    pub fn as_error_message(self: &Output, input: &str) -> String {
        let span = Span::new(input, self.source_ref.pos_start, self.source_ref.pos_end);
        let (line, col) = span.as_ref().unwrap().start_pos().line_col();
        format!("Line {}, Column {}: Error: {}", line, col, self.text)
    }
}

#[derive(Debug)]
pub enum BoundaryCondition<'s> {
    Neumann(Ast<'s>),
    Dirichlet(Ast<'s>),
}

#[derive(Debug)]
pub struct Variable<'s> {
    pub name: &'s str,
    pub dim: usize,
    pub bounds: (f64, f64),
    pub equation: Option<Ast<'s>>,
    pub dependents: Vec<Weak<Variable<'s>>>,
    pub time_index: Option<usize>,
    pub init_conditions: Vec<BoundaryCondition<'s>>,
}

impl<'s> Variable<'s> {
    pub fn is_time_dependent(&self) -> bool {
        self.time_index.is_some()
    }
    pub fn is_state(&self) -> bool {
        self.equation.is_some()
    }
    pub fn new(node: &Box<Ast<'s>>, info: &mut ModelInfo<'s>) -> Variable<'s> {
        match &node.kind {
            AstKind::Unknown(unknown) => {
                let is_time = unknown.name == "t";
                let bounds = match &unknown.codomain {
                    Some(r) => match &r.kind {
                        AstKind::Range(r) => (r.lower, r.upper),
                        AstKind::Name(name) => match *name {
                            "NonNegative" => (0.0, f64::INFINITY),
                            "R" => (-f64::INFINITY, f64::INFINITY),
                            _ => {
                                info.output.push(Output::new(
                                    format!("Unknown domain {}", name),
                                    node.span,
                                ));
                                (-f64::INFINITY, f64::INFINITY)
                            }
                        },
                        _ => unreachable!(),
                    },
                    None => (if is_time { 0.0 } else { -f64::INFINITY }, f64::INFINITY),
                };
                Variable {
                    name: unknown.name,
                    dim: 1,
                    time_index: None,
                    dependents: Vec::new(),
                    bounds,
                    equation: None,
                    init_conditions: Vec::new(),
                }
            }
            AstKind::Definition(dfn) => {
                let deps = dfn.rhs.get_dependents();
                let dependents: Vec<&str> = deps.into_iter().collect();
                let time_index = dependents.iter().position(|d| *d == "t");
                let bounds = (-f64::INFINITY, f64::INFINITY);
                Variable {
                    name: dfn.name,
                    dim: 1,
                    dependents: Vec::new(),
                    bounds,
                    equation: Some(*dfn.rhs),
                    time_index,
                    init_conditions: Vec::new(),
                }
            }
            _ => panic!("Cannot create variable from {}", node),
        }
    }
}

#[derive(Debug)]
pub struct ModelInfo<'s> {
    pub name: &'s str,
    pub variables: HashMap<&'s str, Rc<Variable<'s>>>,
    pub time: Weak<Variable<'s>>,
    pub stmts: Vec<Ast<'s>>,
    pub output: Vec<Output>,
}

impl<'s> ModelInfo<'s> {
    pub fn new(name: &'s str) -> ModelInfo<'s> {
        let time = Rc::new(Variable {
            name: "t",
            dim: 1,
            bounds: (0.0, f64::INFINITY),
            equation: None,
            dependents: Vec::new(),
            time_index: None,
            init_conditions: Vec::new(),
        });
        Self {
            name,
            output: Vec::new(),
            stmts: Vec::new(),
            variables: HashMap::from([(time.name, time)]),
            time: Rc::downgrade(&time),
        }
    }
    pub fn build(name: &'s str, ast: Vec<Box<Ast<'s>>>) -> Result<Self, String> {
        let model_refs: Vec<&Model> = ast.iter().filter_map(|n| AstKind::model(&n.kind)).collect();
        let ast_refs: Vec<&Box<Ast>> = ast.iter().collect();
        match model_refs.iter().position(|v| v.name == name) {
            Some(i) => {
                let other_models = [&model_refs[..i], &model_refs[i..]].concat();
                let other_asts = [&ast_refs[..i], &ast_refs[i..]].concat();
                let mut model_info =
                    Self::builder(&ast[i], model_refs[i], &other_models, &other_asts);
                model_info.allocate_stmts();
                model_info.check_model();
                Ok(model_info)
            }
            None => Err(format!("Model name {} not found", name)),
        }
    }
    fn build_submodel(
        name: &'s str,
        models: &Vec<&ast::Model<'s>>,
        asts: &Vec<&Box<Ast<'s>>>,
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

    fn allocate_stmts(&mut self) {
        for stmt in self.stmts {
            match &stmt.kind {
                AstKind::Submodel(submodel_call) => panic!("Should be no submodels here"),
                AstKind::Equation(eqn) => {
                    // its an dirichlet initial condition if:
                    //  - the lhs is a call with a name equal to one of the variables,
                    //  - that variable has a dependent t,
                    //  - there is a number equal to the lower bound of time in the argument corresponding to time
                    let mut is_ic = false;
                    if let AstKind::Call(Call { fn_name, args }) = &eqn.lhs.kind {
                        if let Some(v) = self.variables.get(fn_name) {
                            if let Some(time_index) = v.time_index {
                                if let AstKind::Number(value) = args[time_index].kind {
                                    if value == self.time.bounds.0 {
                                        is_ic = true;
                                        v.init_conditions
                                            .push(BoundaryCondition::Dirichlet(*eqn.rhs))
                                    }
                                }
                            }
                        }
                    }
                }
                AstKind::RateEquation(reqn) => {
                    match self.variables.get(reqn.name) {
                        Some(v) => {
                            if v.is_state() && v.is_time_dependent() {
                                v.equation = reqn.rhs;
                            } else {
                                info.output.push(Output::new(
                                    format!(
                                        "Rate equation invalid: variable {} does not depend on time",
                                        v.name
                                    ),
                                    stmt.span,
                                ));
                            }
                        }
                        None => info.output.push(Output::new(
                            format!("name {} not found", reqn.name),
                            stmt.span,
                        )),
                    }
                }
                AstKind::Definition(_) => {}
                _ => (),
            }
        }
    }

    fn builder(
        ast: &Box<Ast<'s>>,
        model: &ast::Model<'s>,
        models: &Vec<&ast::Model<'s>>,
        asts: &Vec<&Box<Ast<'s>>>,
    ) -> Self {
        
        let info = Self::new(model.name);
        let reserved = ["u", "dudt", "t", "F", "G", "input"];
        // create variables from unknowns
        for node in model.unknowns.iter() {

            let var = Rc::new(Variable::new(node, &mut info));

            // check its not in list of reserved names
            if var.name == "t" {
                info.time = Rc::downgrade(&var);
            } else if reserved.contains(&var.name) {
                info.output.push(Output::new(
                    format!("Name {} is reserved", var.name),
                    node.span,
                ));
            } else {
                info.variables.insert(var.name, var);
            }
        }
        // set dependents
        for node in model.unknowns.iter() {
            if let AstKind::Unknown(u) = node.kind {
                if let Some(var) = info.variables.get(u.name) {
                    for dep in u.dependents {
                        if let Some(dep_var) = info.variables.get(dep) {
                            var.dependents.push(Rc::downgrade(&dep_var));
                        }
                    }

                }
            }
        }
        for stmt in model.statements.iter() {
            match &stmt.kind {
                AstKind::Submodel(submodel_call) => {
                    // find name in models
                    let mut submodel = match Self::build_submodel(submodel_call.name, models, asts)
                    {
                        Some(x) => x,
                        None => {
                            info.output.push(Output::new(
                                format!("Submodel name {} not found", submodel_call.name),
                                stmt.span,
                            ));
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
                    info.check_expr(&eqn.rhs);
                    info.check_expr(&eqn.lhs);
                    info.stmts.push(*stmt.clone());
                }
                AstKind::RateEquation(reqn) => {
                    // check name exists and variable is state and nonconstant
                    info.check_expr(&reqn.rhs);
                    info.stmts.push(*stmt.clone());
                    
                }
                AstKind::Definition(_) => {
                    let var = Variable::new(&stmt, &mut info);
                    if reserved.contains(&var.name) {
                        info.output.push(Output::new(
                            format!("Name {} is reserved", var.name),
                            stmt.span,
                        ));
                    }
                    info.variables.insert(var.name, Rc::new(var));
                    info.stmts.push(*stmt.clone());
                }
                _ => (),
            }
        }
        info
    }

    fn add_submodel(
        &mut self,
        submodel: & mut ModelInfo<'s>,
        submodel_call: & ast::Submodel<'s>,
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
            self.output.push(Output::new(
                format!(
                    "Model is underdetermined, only {} equations for {} unknowns",
                    n_eqns, n_unknowns
                ),
                self.ast_node.span,
            ));
        } else if n_eqns > n_unknowns {
            self.output.push(Output::new(
                format!(
                    "Model is overdetermined, only {} equations for {} unknowns",
                    n_eqns, n_unknowns
                ),
                self.ast_node.span,
            ));
        }
        // check that variables with derivatives have the right number of boundary conditions
        todo!()
    }

    fn check_expr(&mut self, expr: & Box<Ast<'s>>) {
        match &expr.kind {
            AstKind::Name(name) => {
                // check name exists
                if self.variables.iter().find(|v| v.name == *name).is_none() {
                    self.output.push(Output::new(
                        format!("name {} not found", name),
                        expr.span,
                    ))
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
                        self.output.push(Output::new(
                            format!("Invalid use of variable {}, please use \"{}\" by itself without referring to dependent variables", call.fn_name, call.fn_name),
                            expr.span,
                        ))
                    } else {
                        self.output.push(Output::new(
                            format!("Function or variable {} not found", call.fn_name),
                            expr.span,
                        ))
                    }
                }
                for arg in &call.args {
                    self.check_expr(arg);
                }
            }
            _ => unreachable!(),
        }
    }
    fn find_replacements(
        &mut self,
        submodel: & ModelInfo<'s>,
        submodel_call: &ast::Submodel<'s>,
    ) -> HashMap<&'s str, &Box<Ast<'s>>> {
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
                        self.output.push(Output::new(
                            format!(
                                "Cannot find unknown {} in model {}",
                                name, submodel.name
                            ),
                            arg.span,
                        ));
                    }
                } else {
                    if found_kwarg {
                        self.output.push(Output::new(
                            format!("positional argument found after keyword argument"),
                            arg.span,
                        ));
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
