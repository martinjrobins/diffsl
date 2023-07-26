use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Call;
use crate::ast::Model;
use crate::ast::StringSpan;
use pest::Span;
use std::boxed::Box;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::iter::zip;
use std::rc::Rc;

#[derive(Debug)]
pub struct Output {
    pub text: String,
    pub source_ref: Option<StringSpan>,
    pub secondary_txts: Vec<String>,
    pub secondary_refs: Vec<StringSpan>,
}

impl Output {
    pub fn new(text: String, span: Option<StringSpan>) -> Self {
        Self {
            text: text,
            source_ref: span,
            secondary_txts: Vec::new(),
            secondary_refs: Vec::new(),
        }
    }
    pub fn as_error_message(self: &Output, input: &str) -> String {
        if let Some(source_ref) = self.source_ref {
            let span = Span::new(input, source_ref.pos_start, source_ref.pos_end);
            let (line, col) = span.as_ref().unwrap().start_pos().line_col();
            format!("Line {}, Column {}: Error: {}", line, col, self.text)
        } else {
            format!("Error: {}", self.text)
        }
    }
}

#[derive(Debug)]
pub struct BoundaryCondition<'s> {
    pub variable: Rc<RefCell<Variable<'s>>>,
    pub location: f64,
    pub equation: Ast<'s>,
    pub is_dirichlet:  bool,
}

#[derive(Debug)]
pub struct Variable<'s> {
    pub name: &'s str,
    pub time_gradient_name: String,
    pub dim: usize,
    pub bounds: (f64, f64),
    pub equation: Option<Ast<'s>>,
    pub expression: Option<Ast<'s>>,
    pub dependents: Vec<Rc<RefCell<Variable<'s>>>>,
    pub time_index: Option<usize>,
    pub init_conditions: Vec<BoundaryCondition<'s>>,
}

impl<'a> fmt::Display for Variable<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let deps_disp: Vec<_> = self.dependents.iter().map(|dep| dep.borrow().name).collect();
        if deps_disp.len() > 0 {
            write!(f, "{}({})", self.name, deps_disp.join(","))
        } else {
            write!(f, "{}", self.name)
        }
    }
}


impl<'s> Variable<'s> {
    pub fn is_time_dependent(&self) -> bool {
        self.time_index.is_some() || self.is_time()
        || self.dependents.iter().any(|d| d.borrow().is_time_dependent())
    }
    pub fn is_independent(&self) -> bool {
        self.dependents.is_empty()
    }
    pub fn is_definition(&self) -> bool {
        self.expression.is_some()
    }
    pub fn is_state(&self) -> bool {
        !self.is_definition() && !self.is_independent()
    }
    pub fn has_equation(&self) -> bool {
        self.equation.is_some()
    }
    pub fn has_initial_condition(&self) -> bool {
        self.init_conditions.len() > 0
    }
    pub fn is_time(&self) -> bool {
        return self.name == "t";
    }
    pub fn is_algebraic(&self) -> Option<bool> {
        if let Some(eqn) = &self.equation {
            match &eqn.kind {
                AstKind::Equation(_) => Some(true),
                AstKind::RateEquation(_) => Some(false),
                _ => None,
            }
        } else {
            None
        }
    }
    pub fn is_dependent_on_state(&self) -> bool {
        if self.is_definition() {
            self.dependents.iter().any(|dep|  dep.borrow().is_state())
        } else {
            self.is_state()
        }

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
                                info.errors.push(Output::new(
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
                    time_gradient_name: format!("d{}dt", unknown.name),
                    dim: 1,
                    time_index: None,
                    dependents: Vec::new(),
                    bounds,
                    equation: None,
                    expression: None,
                    init_conditions: Vec::new(),
                }
            }
            AstKind::Definition(dfn) => {
                let bounds = (-f64::INFINITY, f64::INFINITY);
                Variable {
                    name: dfn.name,
                    time_gradient_name: format!("d{}dt", dfn.name),
                    dim: 1,
                    dependents: Vec::new(),
                    bounds,
                    equation: None,
                    expression: Some(dfn.rhs.as_ref().clone()),
                    time_index: None,
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
    pub unknowns: Vec<Rc<RefCell<Variable<'s>>>>,
    pub definitions: Vec<Rc<RefCell<Variable<'s>>>>,
    variables: HashMap<&'s str, Rc<RefCell<Variable<'s>>>>,
    pub time: Rc<RefCell<Variable<'s>>>,
    pub stmts: Vec<Ast<'s>>,
    pub errors: Vec<Output>,
}

impl<'s> ModelInfo<'s> {
    pub fn new(name: &'s str) -> ModelInfo<'s> {
        let t_name = "t";
        let time = Rc::new(RefCell::new(Variable {
            name: t_name,
            time_gradient_name: format!("d{}dt", t_name),
            dim: 1,
            bounds: (0.0, f64::INFINITY),
            equation: None,
            expression: None,
            dependents: Vec::new(),
            time_index: None,
            init_conditions: Vec::new(),
        }));
        Self {
            name,
            errors: Vec::new(),
            unknowns: Vec::new(),
            definitions: Vec::new(),
            stmts: Vec::new(),
            variables: HashMap::from([(t_name, time.clone())]),
            time: time.clone(),
        }
    }
    pub fn build(name: &'s str, ast: &'s Vec<Box<Ast<'s>>>) -> Result<Self, String> {
        let model_refs: Vec<&Model> = ast.iter().filter_map(|n| n.kind.as_model()).collect();
        match model_refs.iter().position(|v| v.name == name) {
            Some(i) => {
                let other_models = [&model_refs[..i], &model_refs[i..]].concat();
                let mut model_info =
                    Self::builder(model_refs[i], &other_models);
                model_info.allocate_stmts(&ast[i]);
                Ok(model_info)
            }
            None => Err(format!("Model name {} not found", name)),
        }
    }
    fn build_submodel(
        name: &'s str,
        models: &Vec<&ast::Model<'s>>,
    ) -> Option<Self> {
        match models.iter().position(|v| v.name == name) {
            Some(i) => {
                let other_models = [&models[..i], &models[i..]].concat();
                Some(Self::builder(
                    models[i],
                    &other_models,
                ))
            }
            None => None,
        }
    }


    fn allocate_stmt<'a>(&'a mut self, stmt: Ast<'s>) -> Option<Ast<'s>> {
        //TODO use if-let chaining
        let bc_opt = match &stmt.kind {
            AstKind::Equation(eqn) => {
                // its an dirichlet initial condition if:
                //  - the lhs is a call with a name equal to one of the variables,
                //  - that variable has a dependent t,
                //  - there is a number equal to the lower bound of time in the argument corresponding to time
                if let AstKind::Call(Call { fn_name, args }) = &eqn.lhs.kind {
                    if let Some(v_cell) = self.variables.get(fn_name) {
                        let v = v_cell.borrow();
                        if let Some(time_index) = v.time_index {
                            if let AstKind::CallArg(call_arg) = &args[time_index].kind {
                                if let AstKind::Number(value) = call_arg.expression.kind {
                                    if value == self.time.borrow().bounds.0 {
                                        Some((
                                            v_cell.clone(),
                                            BoundaryCondition{
                                                    variable: self.time.clone(),
                                                    location: value,
                                                    equation: *eqn.rhs.clone(),
                                                    is_dirichlet: true,
                                            }
                                        ))
                                    } else {
                                        self.errors.push(Output::new(
                                            format!(
                                                "Did you mean to set an initial condition here? equation should be {}({}) = ...",
                                                fn_name, self.time.borrow().bounds.0, 
                                            ),
                                            args[time_index].span,
                                        ));
                                        None
                                    }
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None }
                } else { None }
            },
            _ => None,
        };
        if let Some((v_cell, bc)) = bc_opt {
            v_cell.borrow_mut().init_conditions.push(bc);
            return None;
        }
        let allocated_to = match &stmt.kind {
            AstKind::RateEquation(reqn) => {
                match self.variables.get(reqn.name) {
                    Some(v_c) => {
                        let v = v_c.borrow();
                        if v.is_state() && v.is_time_dependent() {
                            Some(v_c.clone())
                        } else {
                            self.errors.push(Output::new(
                                format!(
                                    "Rate equation invalid: variable {} does not depend on time",
                                    v.name
                                ),
                                stmt.span,
                            ));
                            None
                        }
                    }
                    None => {
                        self.errors.push(Output::new(
                            format!("name {} not found", reqn.name),
                            stmt.span,
                        ));
                        None
                    }
                }
            },
            _ => None,
        };
        if let Some(v_cell) = allocated_to {
            v_cell.borrow_mut().equation = Some(stmt);
            None
        } else {
            Some(stmt)
        }
    }
    fn allocate_stmts(&mut self, ast: &Box<Ast<'s>>) {
        
        // move stmts out of self so we can move them
        let mut stmts: Vec<Ast<'s>> = Vec::new();
        std::mem::swap(& mut self.stmts, & mut stmts);
        let unallocated_eqns: Vec<Ast> = stmts.into_iter().filter_map(|stmt| self.allocate_stmt(stmt)).collect();



        let unallocated_state_vars: Vec<Rc<RefCell<Variable>>> = self.variables.iter()
            .filter_map(|(_name, v)| if v.borrow().is_state() && !v.borrow().has_equation() { Some(v.clone()) } else { None })
            .collect();
        if unallocated_eqns.len() != unallocated_state_vars.len() {
            let msg = if unallocated_state_vars.len() > unallocated_eqns.len() {
                "Model is underdetermined"
            } else {
                "Model is overdetermined"
            };
            let unallocated_eqns_disp: Vec<String> = unallocated_eqns.iter().map(|eqn| eqn.to_string()).collect();
            let unallocated_state_vars_disp: Vec<String> = unallocated_state_vars.iter().map(|var| var.borrow().to_string()).collect();
            self.errors.push(Output::new(
                format!(
                    "{}, {} equations for {} unknowns. Equations are: [{}]. Unknowns are: [{}]",
                    msg,
                    unallocated_eqns.len(), unallocated_state_vars.len(),
                    unallocated_eqns_disp.join(", "), unallocated_state_vars_disp.join(", ")
                ),
                ast.span,
            ));
        } else {
            for (eqn, state_var) in zip(unallocated_eqns, unallocated_state_vars) {
                state_var.borrow_mut().equation = Some(eqn);
            }
        }

        for (_, v_cell) in self.variables.iter() {
            // check all non-algebraic state variables have an initial condition
            let v = v_cell.borrow();
            if v.is_state() && v.has_equation() {
                if !v.is_algebraic().unwrap() && !v.has_initial_condition() {
                    self.errors.push(Output::new(
                        format!("{} does not have an inital condition", v),
                        ast.span,
                    ));
                }
                // check algebraic variables do not have initial conditions 
                if v.is_algebraic().unwrap() && v.has_initial_condition() {
                    self.errors.push(Output::new(
                        format!("overdetermined initial condition, algebraic variable {} should not have an initial condition", v),
                        v.init_conditions[0].equation.span,
                    ));
                }
            }
        }
    }
    

    fn set_dependents(&self, var: & mut Variable<'s>, deps: &Vec<&'s str>) {
        for dep in deps {
            if let Some(dep_var) = self.variables.get(dep) {
                if dep_var.borrow().is_time() {
                    var.time_index = Some(var.dependents.len());
                }
                var.dependents.push(dep_var.clone());
            }
        }
    }

    fn builder(
        model: &ast::Model<'s>,
        models: &Vec<&ast::Model<'s>>,
    ) -> Self {
        
        let mut info = Self::new(model.name);
        let reserved = ["u", "dudt", "t", "F", "G", "input"];
        // create variables from unknowns
        for node in model.unknowns.iter() {

            let var_cell = Rc::new(RefCell::new(Variable::new(node, &mut info)));
            let var = var_cell.borrow();
            info.unknowns.push(var_cell.clone());

            // check its not in list of reserved names
            if var.name == "t" {
                info.time = var_cell.clone();
            } else if reserved.contains(&var.name) {
                info.errors.push(Output::new(
                    format!("Name {} is reserved", var.name),
                    node.span,
                ));
            } else {
                info.variables.insert(var.name, var_cell.clone());
            }
        }
        // set dependents
        for node in model.unknowns.iter() {
            if let AstKind::Unknown(u) = &node.kind {
                if let Some(var) = info.variables.get(u.name) {
                    info.set_dependents(& mut var.borrow_mut(), &u.dependents);
                    // if time is a dependent then add to outputs
                }
            }
        }
        for stmt in model.statements.iter() {
            match &stmt.kind {
                AstKind::Submodel(submodel_call) => {
                    // find name in models
                    let mut submodel = match Self::build_submodel(submodel_call.name, models)
                    {
                        Some(x) => x,
                        None => {
                            info.errors.push(Output::new(
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
                    info.check_expr(&eqn.rhs);
                    info.check_expr(&eqn.lhs);
                    info.stmts.push(*stmt.clone());
                }
                AstKind::RateEquation(reqn) => {
                    // check name exists and variable is state and nonconstant
                    info.check_expr(&reqn.rhs);
                    info.stmts.push(*stmt.clone());
                    
                }
                AstKind::Definition(dfn) => {
                    let var_cell = Rc::new(RefCell::new(Variable::new(&stmt, &mut info)));
                    let mut var = var_cell.borrow_mut();
                    if reserved.contains(&var.name) {
                        info.errors.push(Output::new(
                            format!("Name {} is reserved", var.name),
                            stmt.span,
                        ));
                    }
                    let deps = dfn.rhs.get_dependents();
                    let dependents: Vec<&str> = deps.into_iter().collect();
                    var.time_index = dependents.iter().position(|d| *d == "t");
                    info.set_dependents(& mut var, &dependents);
                    info.definitions.push(var_cell.clone());
                    info.variables.insert(var.name, var_cell.clone());
                    info.check_expr(&dfn.rhs);
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
        self.errors.append(&mut submodel.errors);
        let replacements = self.find_replacements(submodel, submodel_call);

        // add all the stmts with replacement
        for stmt in &submodel.stmts {
            self.stmts.push(stmt.clone_and_subst(&replacements));
        }

        // add all the definitions with replacement
        for (name, var_cell) in &submodel.variables {
            let var_cell = var_cell.clone();
            if !var_cell.borrow().is_definition() {
                continue;
            }
            // apply replacements to equation
            {
                let mut var = var_cell.borrow_mut(); 
                if let Some(eqn) = &var.equation {
                    var.equation = Some(eqn.clone_and_subst(&replacements));

                }
            }
            self.variables.insert(name, var_cell);
        }
    }

    fn check_expr(&mut self, expr: & Box<Ast<'s>>) {
        match &expr.kind {
            AstKind::Name(name) => {
                // check name exists
                if self.variables.iter().find(|(var_name, _)| *var_name == name).is_none() {
                    self.errors.push(Output::new(
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
                let functions = ["sin" , "cos" , "tan" , "pow" , "exp" , "log" , "sqrt" , "abs"];
                if functions.contains(&call.fn_name) {
                    // built in functions all have 1 arg
                    // built in functions should have no keyword args
                    if call.args.len() != 1 {
                        self.errors.push(Output::new(
                            format!("incorrect number of given arguments ({} instead of {}) for function {}", call.args.len(), 1, call.fn_name),
                            expr.span,
                        ));
                    }
                    for arg in call.args.iter() {
                        if let AstKind::CallArg(call_arg) = &arg.kind {
                            if let Some(_) = call_arg.name {
                                self.errors.push(Output::new(
                                    format!("keyword args not allowed for built-in funcitons"),
                                    arg.span,
                                ));
                            }
                        } else {
                            panic!("all args should be CallArgs")
                        }
                    }
                } else if let Some((_, var_cell)) = self.variables.iter().find(|(_name, var)| var.borrow().name == call.fn_name) {
                    // variable call, check we've got all the right call args
                    let var = var_cell.borrow();
                    if var.dependents.len() != call.args.len() {
                        self.errors.push(Output::new(
                            format!("incorrect number of arguments ({}) for dependent variable {}", call.args.len(), var),
                            expr.span,
                        ));
                    }
                    let mut has_kwarg = false;
                    for arg in call.args.iter() {
                        if let AstKind::CallArg(call_arg) = &arg.kind {
                            if let Some(name) = call_arg.name {
                                has_kwarg = true;
                                if var.dependents.iter().all(|v| v.borrow().name != name) {
                                    self.errors.push(Output::new(
                                        format!("named arg {} does not exist in variable {}", name, var),
                                        arg.span,
                                    ));
                                }
                            } else {
                                if has_kwarg {
                                    self.errors.push(Output::new(
                                        format!("indexed call arg found after named arg"),
                                        arg.span,
                                    ));
                                }
                            }
                        } else {
                            panic!("all args should be CallArgs")
                        }

                    }
                } else {
                    self.errors.push(Output::new(
                        format!("Function or variable {} not found", call.fn_name),
                        expr.span,
                    ));
                }

                // check args
                for arg in &call.args {
                    self.check_expr(arg);
                }
            },
            AstKind::CallArg(arg) => {
                self.check_expr(&arg.expression);
            }
            AstKind::Number(_) => (),
            _ => unreachable!(),
        }
    }
    fn find_replacements<'a>(
        &mut self,
        submodel: & ModelInfo<'s>,
        submodel_call: &'a ast::Submodel<'s>
    ) -> HashMap<&'s str, &'a Box<Ast<'s>>> {
        let mut replacements = HashMap::new();
        let mut found_kwarg = false;

        // find all the replacements specified in the call arguments
        for (i, arg) in submodel_call.args.iter().enumerate() {
            if let AstKind::CallArg(call_arg) = &arg.kind {
                if let Some(name) = call_arg.name {
                    found_kwarg = true;
                    if let Some(_) = submodel.variables.iter().find(|(name, var)| var.borrow().name == **name) {
                        replacements.insert(name, &call_arg.expression);
                    } else {
                        self.errors.push(Output::new(
                            format!(
                                "Cannot find unknown {} in model {}",
                                name, submodel.name
                            ),
                            arg.span,
                        ));
                    }
                } else {
                    if found_kwarg {
                        self.errors.push(Output::new(
                            format!("positional argument found after keyword argument"),
                            arg.span,
                        ));
                    }
                    replacements.insert(submodel.unknowns[i].borrow().name, &call_arg.expression);
                };
            }
        }
        replacements
    }
}

#[cfg(test)]
mod tests {
    use crate::{builder::ModelInfo, ms_parser::parse_string};

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
        assert_eq!(model_info.variables.len(), 3);
        assert!(model_info.variables.get("i").is_some());
        assert!(model_info.variables.get("t").is_some());
        assert_eq!(model_info.stmts.len(), 0);
        assert_eq!(model_info.errors.len(), 0);
    }
    #[test]
    fn rate_equation() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t) ) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.variables.len(), 4);
        assert_eq!(model_info.errors.len(), 0);
    }
    #[test]
    fn init_cond_wrong_lower_bound() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t) ) { 
            dot(y) = r * y * (1 - y / k)
            y(1) = 1.0
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert!(model_info.errors.len() > 0);
        assert!(model_info.errors.iter().any(|o| o.as_error_message(text).contains("Did you mean to set an initial condition")));
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
        assert_eq!(model_info.variables.len(), 4);
        assert_eq!(model_info.stmts.len(), 0);
        for o in model_info.errors.iter() {
            println!("{}", o.as_error_message(text));
        }
        assert_eq!(model_info.errors.len(), 1);
        assert!(model_info.errors[0].as_error_message(text).contains("does not have an inital condition"));
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
        assert_eq!(model_info.variables.len(), 3);
        assert_eq!(model_info.errors.len(), 2);
        assert!(model_info.errors[0].text.contains("resistorr") == true);
        assert!(model_info.errors[1].text.contains("underdetermined") == true);
    }
    #[test]
    fn submodel_replacements() {
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
        assert_eq!(model_info.variables.len(), 3);
        assert_eq!(model_info.errors.len(), 0);
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
        assert_eq!(model_info.variables.len(), 4);
        assert_eq!(model_info.errors.len(), 2);
        assert!(model_info.errors[0].text.contains("doesnotexist") == true);
        assert!(model_info.errors[1].text.contains("underdetermined") == true);
    }
    #[test]
    fn alg_variable_with_ic() {
        let text = "
        model resistor(i(t)) {
            0 = i 
            i(0) = 1
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("resistor", &models).unwrap();
        assert_eq!(model_info.errors.len(), 1);
        assert!(model_info.errors[0].text.contains("overdetermined initial condition") == true);
    }
}
