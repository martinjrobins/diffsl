use anyhow::Result;
use core::panic;
use std::cell::RefCell;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use itertools::chain;
use ndarray::Array1;

use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Call;
use crate::ast::StringSpan;
use crate::builder::ModelInfo;
use crate::builder::Variable;
use crate::error::ValidationError;
use crate::error::ValidationErrors;

//TODO: inputs, states and tensors share a lot of code, refactor into a trait

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct TensorBlock<'s> {
    start: Index,
    shape: Shape,
    expr: Ast<'s>,
}

impl<'s> TensorBlock<'s> {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn start(&self) -> &Index {
        &self.start
    }

    pub fn expr(&self) -> &Ast<'s> {
        &self.expr
    }
    
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl<'s> fmt::Display for TensorBlock<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct Tensor<'s> {
    name: &'s str,
    shape: Shape,
    elmts: Vec<TensorBlock<'s>>,
    indices: Vec<char>,
}

impl<'s> Tensor<'s> {
    pub fn new(name: &'s str) -> Self {
        Self {
            name,
            shape: Shape::zeros(0),
            elmts: Vec::new(),
            indices: Vec::new(),
        }
    }
    pub fn new_binop(name: &'s str, n_states: usize, lhs: &'s str, rhs: &'s str, op: char) -> Self {
        Self {
            name,
            shape: Shape::from_vec(vec![n_states]),
            indices: vec!['i'],
            elmts: vec![TensorBlock {
                start: Index::from_vec(vec![0]),
                shape: Shape::from_vec(vec![n_states]),
                expr: Ast {
                    kind: AstKind::new_binop(
                        op,
                        Ast {
                            kind: AstKind::new_name(lhs),
                            span: None,
                        },
                        Ast {
                            kind: AstKind::new_name(rhs),
                            span: None,
                        },
                    ),
                    span: None,
                },
            }],
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn elmts(&self) -> &[TensorBlock] {
        self.elmts.as_ref()
    }
    
    pub fn from_vec(name: &'s str, elmts: Vec<TensorBlock<'s>>, indices: Vec<char>) -> Self {
        let rank = elmts.iter().fold(0, |acc, i| max(acc, i.rank()));
        let shape = elmts.iter().fold(Shape::zeros(rank), |mut acc, i| {
            let max_index = i.start().mapv(|x| usize::try_from(x).unwrap()) + i.shape();
            for i in 0..acc.shape()[0] {
                acc[i] = max(acc[i], max_index[i]);
            }
            acc
        });
        Self {
            name,
            shape,
            elmts,
            indices,
        }
    }
}

impl<'s> fmt::Display for Tensor<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.indices.len() > 0 {
            write!(f, "{}_", self.name).and_then(|_| self.indices.iter().fold(Ok(()), |acc, i| {
                acc.and_then(|_| write!(f, "{}", i))
            }))
        } else {
            write!(f, "{}", self.name)
        }.and_then(|_|  write!(f, " {{\n"))
        .and_then(|_| self.elmts.iter().fold(Ok(()), |acc, e| {
            acc.and_then(|_| write!(f, "  {},\n", e))
        }))
        .and_then(|_| write!(f, "}}"))
    }
}

#[derive(Debug)]
pub struct Inputs<'s> {
    shape: Shape,
    elmts: Vec<Input<'s>>,
    indices: Vec<char>,
}

impl<'s> Inputs<'s> {
    pub fn new() -> Self {
        Self {
            shape: Shape::zeros(0),
            elmts: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn elmts(&self) -> &[Input] {
        self.elmts.as_ref()
    }
    pub fn from_vec(inputs: Vec<Input<'s>>) -> Self {
        if inputs.is_empty() {
            return Self::new();
        }
        let last = inputs.last().unwrap();
        let shape = last.start().mapv(|x| usize::try_from(x).unwrap()) + last.shape();
        if shape.len() > 1 {
            panic!("Inputs must be scalar or 1D");
        }
        let indices = if shape.len() == 0 || shape[0] == 1 { 
            vec![]
        } else {
            vec!['i']
        };
        Self {
            shape,
            elmts: inputs,
            indices,
        }
    }
}

impl<'s> fmt::Display for Inputs<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.elmts.len() > 1 {
            write!(f, "in_").and_then(|_| self.indices.iter().fold(Ok(()), |acc, i| {
                acc.and_then(|_| write!(f, "{}", i))
            }))
        } else {
            write!(f, "in")
        }
        .and_then(|_| write!(f, " {{\n"))
        .and_then(|_| self.elmts.iter().fold(Ok(()), |acc, e| {
            acc.and_then(|_| write!(f, "  {},\n", e))
        }))
        .and_then(|_| write!(f, "}}"))
    }
}

#[derive(Debug)]
// the p[i] in F(t, p, u, u_dot) = G(t, p, u)
pub struct Input<'s> {
    name: &'s str,
    start: Index,
    shape: Shape,
    domain: (f64, f64),
}

impl<'s> Input<'s> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn start(&self) -> &Index {
        &self.start
    }

    pub fn domain(&self) -> (f64, f64) {
        self.domain
    }

    pub fn domain_str(&self) -> String {
        if self.domain.0 == f64::NEG_INFINITY && self.domain.1 == f64::INFINITY {
            return "R".to_string();
        } else {
            format!("[{}, {}]", self.domain.0, self.domain.1)
        }
    }
}

impl<'s> fmt::Display for Input<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
        .and_then(|_| write!(f, " -> {}", self.domain_str()))
        .and_then(|_| if self.shape[0] > 1 {
            write!(f, " ** {}", self.shape[0])
        } else {
            Ok(())
        })
    }
}

#[derive(Debug)]
pub struct States<'s> {
    shape: Shape,
    elmts: Vec<State<'s>>,
    indices: Vec<char>,
}

impl<'s> States<'s> {
    pub fn new() -> Self {
        Self {
            shape: Shape::zeros(0),
            elmts: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn elmts(&self) -> &[State] {
        self.elmts.as_ref()
    }

    pub fn indices(&self) -> &[char] {
        self.indices.as_ref()
    }
    pub fn from_vec(elmts: Vec<State<'s>>, indices: Vec<char>) -> Self {
        if elmts.is_empty() {
            return Self::new();
        }
        let last = elmts.last().unwrap();
        let shape = last.start().mapv(|x| usize::try_from(x).unwrap()) + last.shape();
        Self {
            shape: shape,
            elmts: elmts,
            indices: indices,
        }
    }
    pub fn get_init(&self) -> (Tensor<'s>, Tensor<'s>) {
        let alg_init = Ast {
            kind: AstKind::Number(0.),
            span: None,
        };
        let mut u0elmts = Vec::new();
        let mut dotu0elmts = Vec::new();
        let mut index = Index::zeros(1);
        for s in &self.elmts {
            let expr = match &s.init {
                Some(eq) => eq.clone(),
                None => alg_init.clone(),
            };
            if s.shape().len() != 1 {
                panic!("state shape must be 1D");
            }
            u0elmts.push(TensorBlock {
                expr,
                start: index.clone(),
                shape: s.shape().clone(),
            });
            dotu0elmts.push(TensorBlock {
                expr: alg_init.clone(),
                start: index.clone(),
                shape: s.shape().clone(),
            });
            index = index + s.shape().mapv(|x| i64::try_from(x).unwrap());
        }
        let shape = index.mapv(|x| usize::try_from(x).unwrap());
        (
            Tensor {
                name: "u0",
                shape: shape.clone(),
                indices: self.indices.clone(),
                elmts: u0elmts,
            },
            Tensor {
                name: "dotu0",
                shape,
                indices: self.indices.clone(),
                elmts: dotu0elmts,
            },
        )
    }
}

impl<'s> fmt::Display for States<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.elmts.len() > 1 {
            write!(f, "u_").and_then(|_| self.indices.iter().fold(Ok(()), |acc, i| {
                acc.and_then(|_| write!(f, "{}", i))
            }))
        } else {
            write!(f, "u")
        }.and_then(|_| write!(f, " {{\n"))
        .and_then(|_| self.elmts.iter().fold(Ok(()), |acc, e| {
            acc.and_then(|_| write!(f, "  {},\n", e))
        }))
        .and_then(|_| write!(f, "}}"))
    }
}


#[derive(Debug)]
// the p[i] in F(t, p, u, u_dot) = G(t, p, u)
pub struct State<'s> {
    name: &'s str,
    start: Index,
    shape: Shape,
    init: Option<Ast<'s>>,
    domain: (f64, f64),
    is_algebraic: bool,
}

impl<'s> State<'s> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn start(&self) -> &Index {
        &self.start
    }

    pub fn init(&self) -> Option<&Ast> {
        self.init.as_ref()
    }

    pub fn is_algebraic(&self) -> bool {
        self.is_algebraic
    }

    pub fn domain(&self) -> (f64, f64) {
        self.domain
    }
    
    pub fn domain_str(&self) -> String {
        if self.domain.0 == f64::NEG_INFINITY && self.domain.1 == f64::INFINITY {
            return "R".to_string();
        } else {
            format!("[{}, {}]", self.domain.0, self.domain.1)
        }
    }
}

impl<'s> fmt::Display for State<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
        .and_then(|_| write!(f, " -> {}", self.domain_str()))
        .and_then(|_| if self.shape[0] > 1 {
            write!(f, " ** {}", self.shape[0])
        } else {
            Ok(())
        })
        .and_then(|_| if self.init.is_some() {
            write!(f, " = {}", self.init.as_ref().unwrap())
        } else {
            Ok(())
        })
    }
}

pub type Shape = Array1<usize>;
pub type Index = Array1<i64>;

struct EnvVar {
    shape: Shape,
    is_time_dependent: bool,
    is_state_dependent: bool,
    is_algebraic: bool,
}

impl EnvVar {
    fn is_time_dependent(&self) -> bool {
        self.is_time_dependent
    }

    fn is_state_dependent(&self) -> bool {
        self.is_state_dependent
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn is_algebraic(&self) -> bool {
        self.is_algebraic
    }
}

struct Env<'s> {
    errs: ValidationErrors,
    vars: HashMap<&'s str, EnvVar>,
}

pub fn broadcast_shapes(shapes: &[&Shape]) -> Option<Shape> {
    if shapes.is_empty() {
        return None;
    }
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap();
    let mut shape = Shape::zeros(max_rank);
    for i in (0..max_rank).rev() {
        let (mdim, compatible) = shapes.iter().map(|s| s.get(i).unwrap_or(&1)).fold(
            (1, true),
            |(mdim, _result), dim| {
                let new_mdim = max(mdim, *dim);
                (new_mdim, *dim == 1 || *dim == new_mdim)
            },
        );
        if !compatible {
            return None;
        }
        shape[i] = mdim;
    }
    Some(shape)
}

impl<'s> Env<'s> {
    pub fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert(
            "t",
            EnvVar {
                shape: Shape::ones(1),
                is_time_dependent: true,
                is_state_dependent: false,
                is_algebraic: true,
            },
        );
        Env {
            errs: ValidationErrors::new(),
            vars,
        }
    }
    pub fn is_tensor_time_dependent(&self, tensor: &Tensor) -> bool {
        tensor.elmts.iter().any(|block| {
            block
                .expr
                .get_dependents()
                .iter()
                .any(|&dep| dep == "t" || self.vars[dep].is_time_dependent())
        })
    }
    pub fn is_tensor_state_dependent(&self, tensor: &Tensor) -> bool {
        tensor.elmts.iter().any(|block| {
            block
                .expr
                .get_dependents()
                .iter()
                .any(|&dep| dep == "u" || self.vars[dep].is_state_dependent())
        })
    }

    pub fn push_var(&mut self, var: &Tensor<'s>) {
        self.vars.insert(
            var.name,
            EnvVar {
                is_algebraic: true,
                shape: var.shape.clone(),
                is_time_dependent: self.is_tensor_time_dependent(var),
                is_state_dependent: self.is_tensor_state_dependent(var),
            },
        );
    }

    pub fn push_state(&mut self, state: &State<'s>) {
        self.vars.insert(
            state.name,
            EnvVar {
                is_algebraic: state.is_algebraic,
                shape: state.shape.clone(),
                is_time_dependent: true,
                is_state_dependent: true,
            },
        );
    }

    pub fn push_input(&mut self, input: &Input<'s>) {
        self.vars.insert(
            input.name,
            EnvVar {
                is_algebraic: true,
                shape: input.shape.clone(),
                is_time_dependent: false,
                is_state_dependent: false,
            },
        );
    }

    fn get(&self, name: &str) -> Option<&EnvVar> {
        self.vars.get(name)
    }
    fn get_shape_binary_op(
        &mut self,
        left: &Ast<'s>,
        right: &Ast<'s>,
        indices: &Vec<char>,
    ) -> Option<Shape> {
        let left_shape = self.get_shape(left, indices)?;
        let right_shape = self.get_shape(right, indices)?;
        match broadcast_shapes(&[&left_shape, &right_shape]) {
            Some(shape) => Some(shape),
            None => {
                self.errs.push(ValidationError::new(
                    format!(
                        "cannot broadcast operands together. lhs {} and rhs {}",
                        left_shape, right_shape
                    ),
                    left.span,
                ));
                None
            }
        }
    }
    fn get_shape_dot(
        &mut self,
        call: &Call<'s>,
        ast: &Ast<'s>,
        indices: &Vec<char>,
    ) -> Option<Shape> {
        if call.args.len() != 1 {
            self.errs.push(ValidationError::new(
                format!(
                    "time derivative dot call expects 1 argument, got {}",
                    call.args.len()
                ),
                ast.span,
            ));
            return None;
        }
        let arg = &call.args[0];
        let arg_shape = self.get_shape(arg, indices)?;
        if arg_shape.len() != 1 {
            self.errs.push(ValidationError::new(
                format!(
                    "time derivative dot call expects 1D argument, got {}D",
                    arg_shape.len()
                ),
                ast.span,
            ));
            return None;
        }
        let depends_on = arg.get_dependents();
        // for each state variable, set is_algebraic to false
        for dep in depends_on {
            if let Some(var) = self.vars.get_mut(dep) {
                var.is_algebraic = false;
            }
        }
        return Some(arg_shape);
    }

    fn get_shape_name(
        &mut self,
        name: &str,
        ast: &Ast,
        rhs_indices: &Vec<char>,
        lhs_indices: &Vec<char>,
    ) -> Option<Shape> {
        let var = self.get(name);
        if var.is_none() {
            self.errs.push(ValidationError::new(
                format!("cannot find variable {}", name),
                ast.span,
            ));
            return None;
        }
        let var = var.unwrap();
        let shape = var.shape();

        // work out required number of indices
        let min_rank = {
            let mut i = shape.len();
            loop {
                if i == 0 || shape[i - 1] != 1 {
                    break;
                }
                i -= 1;
            }
            i
        };
        if rhs_indices.len() < min_rank {
            self.errs.push(ValidationError::new(
                format!(
                    "cannot index variable {} with {} indices. Expected at least {} indices",
                    name,
                    rhs_indices.len(),
                    shape.len()
                ),
                ast.span,
            ));
            return None;
        }
        let mut new_shape = Shape::ones(shape.len());
        for (rhs_index, c) in rhs_indices.iter().enumerate() {
            if let Some(lhs_index) = lhs_indices.iter().position(|&x| x == *c) {
                new_shape[lhs_index] = shape[rhs_index];
            } else {
                self.errs.push(ValidationError::new(
                    format!("cannot find index {} in LHS indices {:?}", c, lhs_indices),
                    ast.span,
                ));
                return None;
            }
        }
        Some(new_shape)
    }

    fn get_shape_sum(
        &mut self,
        call: &Call<'s>,
        ast: &Ast<'s>,
        indices: &Vec<char>,
    ) -> Option<Shape> {
        if call.args.len() != 2 {
            self.errs.push(ValidationError::new(
                format!("sum must have 2 arguments. found {}", call.args.len()),
                ast.span,
            ));
            return None;
        }
        if call.args[0].kind.as_name().is_none() {
            self.errs.push(ValidationError::new(
                format!(
                    "sum must have a variable as the first argument. found {}",
                    call.args[0]
                ),
                ast.span,
            ));
            return None;
        }
        let name = call.args[0].kind.as_name().unwrap();
        if name.len() != 1 {
            self.errs.push(ValidationError::new(
                format!(
                    "sum must have a single character variable as the first argument. found {}",
                    name
                ),
                ast.span,
            ));
            return None;
        }
        let index = name.chars().next().unwrap();
        let mut within_sum_indices = indices.clone();
        within_sum_indices.push(index);
        self.get_shape(call.args[1].as_ref(), &within_sum_indices)
    }

    fn get_shape_call(&mut self, call: &Call<'s>, ast: &Ast, indices: &Vec<char>) -> Option<Shape> {
        let shapes = call
            .args
            .iter()
            .map(|c| self.get_shape(c, indices))
            .collect::<Option<Vec<Shape>>>()?;
        match broadcast_shapes(shapes.iter().collect::<Vec<&Shape>>().as_slice()) {
            Some(shape) => Some(shape),
            None => {
                let shape_strs: Vec<String> = shapes.iter().map(|s| s.to_string()).collect();
                self.errs.push(ValidationError::new(
                    format!(
                        "cannot broadcast operands together. shapes {:?}",
                        shape_strs
                    ),
                    ast.span,
                ));
                None
            }
        }
    }

    pub fn get_shape(&mut self, ast: &Ast<'s>, indices: &Vec<char>) -> Option<Shape> {
        match &ast.kind {
            AstKind::Parameter(p) => self.get_shape(&p.domain, indices),
            AstKind::Binop(binop) => {
                self.get_shape_binary_op(binop.left.as_ref(), binop.right.as_ref(), indices)
            }
            AstKind::Monop(monop) => self.get_shape(monop.child.as_ref(), indices),
            AstKind::Call(call) => match call.fn_name {
                "sum" => self.get_shape_sum(&call, ast, indices),
                "dot" => self.get_shape_dot(&call, ast, indices),
                _ => self.get_shape_call(&call, ast, indices),
            },
            AstKind::CallArg(arg) => self.get_shape(arg.expression.as_ref(), indices),
            AstKind::Index(i) => {
                self.get_shape_binary_op(i.left.as_ref(), i.right.as_ref(), indices)
            }
            AstKind::Slice(s) => {
                self.get_shape_binary_op(s.lower.as_ref(), s.upper.as_ref(), indices)
            }
            AstKind::Number(_) => Some(Shape::ones(1)),
            AstKind::Integer(_) => Some(Shape::ones(1)),
            AstKind::Domain(d) => Some(Shape::ones(d.dim)),
            AstKind::IndexedName(name) => {
                self.get_shape_name(name.name, ast, &name.indices, indices)
            }
            AstKind::Name(name) => self.get_shape_name(name, ast, &vec![], indices),
            _ => panic!("unrecognised ast node {:#?}", ast.kind),
        }
    }
}

#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'s> {
    pub name: &'s str,
    pub lhs: Tensor<'s>,
    pub rhs: Tensor<'s>,
    pub out: Tensor<'s>,
    pub time_indep_defns: Vec<Tensor<'s>>,
    pub time_dep_defns: Vec<Tensor<'s>>,
    pub state_dep_defns: Vec<Tensor<'s>>,
    pub inputs: Inputs<'s>,
    pub states: States<'s>,
}

impl<'s, 'a> fmt::Display for DiscreteModel<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}\n", self.inputs)
        .and_then(|_| self.time_indep_defns.iter().fold(Ok(()), |acc, defn| acc.and_then(|_| write!(f, "{}\n", defn))))
        .and_then(|_| self.time_dep_defns.iter().fold(Ok(()), |acc, defn| acc.and_then(|_| write!(f, "{}\n", defn))))
        .and_then(|_| write!(f, "{}\n", self.states))
        .and_then(|_| self.state_dep_defns.iter().fold(Ok(()), |acc, defn| acc.and_then(|_| write!(f, "{}\n", defn))))
        .and_then(|_| write!(f, "{}\n", self.lhs))
        .and_then(|_| write!(f, "{}\n", self.rhs))
        .and_then(|_| write!(f, "{}\n", self.out))
    }
}

impl<'s> DiscreteModel<'s> {
    pub fn new(name: &'s str) -> Self {
        Self {
            name,
            lhs: Tensor::new("F"),
            rhs: Tensor::new("G"),
            out: Tensor::new("out"),
            time_indep_defns: Vec::new(),
            time_dep_defns: Vec::new(),
            state_dep_defns: Vec::new(),
            inputs: Inputs::new(),
            states: States::new(),
        }
    }
    
    fn build_domain(domain_ast: &Ast<'s>, env: &mut Env<'s>) -> (f64, f64) {
        let domain = domain_ast.kind.as_domain().unwrap();
        match &domain.range.kind {
            AstKind::Range(r) => (r.lower, r.upper),
            AstKind::Name(name) => match *name {
                "R" => (-f64::INFINITY, f64::INFINITY),
                _ => {
                    env.errs.push(ValidationError::new(
                        format!("Unknown domain {}", name),
                        domain_ast.span,
                    ));
                    (-f64::INFINITY, f64::INFINITY)
                }
            },
            _ => {
                env.errs.push(ValidationError::new(
                    format!("Unknown domain, should be a range or a name"),
                    domain_ast.span,
                ));
                (-f64::INFINITY, f64::INFINITY)
            }
        }
    }

    fn build_states(tensor: &ast::Tensor<'s>, env: &mut Env<'s>) -> States<'s> {
        let mut ret = Vec::new();
        let rank = tensor.indices.len();
        if rank == 0 && tensor.elmts.len() > 1 {
            env.errs.push(ValidationError::new(
                format!("cannot have more than one element in a scalar"),
                tensor.elmts[1].span,
            ));
        }
        assert_eq!(tensor.name, "u");
        let mut start = Index::zeros(rank);
        for a in &tensor.elmts {
            if let Some(elmt_shape) = env.get_shape(a.as_ref(), &tensor.indices) {
                match &a.kind {
                    AstKind::Parameter(p) => {
                        let name = p.name;
                        let shape = elmt_shape;
                        if shape.len() > 1 {
                            env.errs.push(ValidationError::new(
                                format!(
                                    "state {} has shape {}, expected scalar or 1D array",
                                    name, shape
                                ),
                                a.span,
                            ));
                        }
                        let domain = Self::build_domain(&p.domain, env);
                        let init = match p.init {
                            Some(ref init) => {
                                if let Some(init_shape) = env.get_shape(init, &tensor.indices) {
                                    if init_shape != shape {
                                        env.errs.push(
                                            ValidationError::new(
                                                format!("state {} has shape {}, but initial value has shape {}", name, shape, init_shape),
                                                a.span
                                            )
                                        );
                                    }
                                }
                                Some(*init.clone())
                            }
                            None => None,
                        };
                        let state = State {
                            name,
                            shape: shape.clone(),
                            init,
                            domain,
                            start: start.clone(),
                            is_algebraic: false,
                        };
                        env.push_state(&state);
                        ret.push(state);
                        start += &shape.mapv(|x| i64::try_from(x).unwrap());
                    }
                    _ => {
                        env.errs.push(ValidationError::new(
                            format!("expected assignment in state definition"),
                            a.span,
                        ));
                    }
                }
            }
        }
        States::from_vec(ret, (&tensor.indices).clone())
    }

    fn build_inputs(tensor: &ast::Tensor<'s>, env: &mut Env<'s>) -> Inputs<'s> {
        let mut ret = Vec::new();
        assert_eq!(tensor.name, "in");

        let rank = tensor.indices.len();
        if rank == 0 && tensor.elmts.len() > 1 {
            env.errs.push(ValidationError::new(
                format!("cannot have more than one element in a scalar"),
                tensor.elmts[1].span,
            ));
        }

        let mut start = Index::zeros(rank);
        for a in &tensor.elmts {
            if let Some(elmt_shape) = env.get_shape(a.as_ref(), &tensor.indices) {
                match &a.kind {
                    AstKind::Parameter(p) => {
                        let name = p.name;
                        let shape = elmt_shape;
                        if shape.len() > 1 {
                            env.errs.push(ValidationError::new(
                                format!("input shape must be a scalar or 1D vector"),
                                a.span,
                            ));
                        }
                        let domain = Self::build_domain(&p.domain, env);
                        let input = Input {
                            name,
                            shape: shape.clone(),
                            domain,
                            start: start.clone(),
                        };
                        env.push_input(&input);
                        ret.push(input);
                        start += &shape.mapv(|x| i64::try_from(x).unwrap());
                    }
                    _ => {
                        env.errs.push(ValidationError::new(
                            format!("expected parameter in input definition"),
                            a.span,
                        ));
                    }
                }
            }
        }
        Inputs::from_vec(ret)
    }

    fn build_array(array: &ast::Tensor<'s>, env: &mut Env<'s>) -> Option<Tensor<'s>> {
        let rank = array.indices.len();
        let mut ret = Vec::new();
        let mut start = Index::zeros(rank);
        if rank == 0 && array.elmts.len() > 1 {
            env.errs.push(ValidationError::new(
                format!("cannot have more than one element in a scalar"),
                array.elmts[1].span,
            ));
        } else if rank > 2 && array.elmts.len() > 1 {
            env.errs.push(ValidationError::new(
                format!("cannot have more than one element in a tensor with rank > 2"),
                array.elmts[1].span,
            ));
        }
        for a in &array.elmts {
            if let AstKind::Parameter(_) = a.kind {
                env.errs.push(ValidationError::new(
                    format!("cannot have parameters in tensor definition"),
                    a.span,
                ));
            }
            if let Some(mut elmt_shape) = env.get_shape(a.as_ref(), &array.indices) {
                if rank == 0 && elmt_shape.len() == 1 {
                    if elmt_shape[0] > 1 {
                        env.errs.push(ValidationError::new(
                            format!("cannot assign an expression with rank > 1 to a scalar, rhs has shape {}", elmt_shape),
                            a.span,
                        ));
                    }
                    // convert to scalar array
                    elmt_shape = vec![].into();
                }
                ret.push(TensorBlock {
                    expr: *a.clone(),
                    start: start.clone(),
                    shape: elmt_shape.clone(),
                });
                start += &elmt_shape.mapv(|x| i64::try_from(x).unwrap());
            }
        }
        let tensor = Tensor::from_vec(array.name, ret, (&array.indices).clone());
        env.push_var(&tensor);
        Some(tensor)
    }

    pub fn build(name: &'s str, ast: &'s Vec<Box<Ast<'s>>>) -> Result<Self, ValidationErrors> {
        let mut env = Env::new();
        let mut ret = Self::new(name);
        let mut read_state = false;
        let mut read_f = false;
        let mut read_g = false;
        let mut read_out = false;
        for (i, tensor_ast) in ast.iter().enumerate() {
            match tensor_ast.kind.as_array() {
                None => env.errs.push(ValidationError::new(
                    "not an array".to_string(),
                    tensor_ast.span,
                )),
                Some(tensor) => {
                    let span = tensor_ast.span;
                    // first array must be in
                    if i == 0 && tensor.name != "in" {
                        env.errs.push(ValidationError::new(
                            "first array must be 'in'".to_string(),
                            span,
                        ));
                    }
                    match tensor.name {
                        "in" => {
                            if tensor.indices.len() > 1 {
                                env.errs.push(ValidationError::new(
                                    "input must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                            ret.inputs = Self::build_inputs(tensor, &mut env);
                        }
                        "u" => {
                            if tensor.indices.len() > 1 {
                                env.errs.push(ValidationError::new(
                                    "u must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                            read_state = true;
                            ret.states = Self::build_states(tensor, &mut env);
                        }
                        "F" => {
                            read_f = true;
                            if tensor.indices.len() > 1 {
                                env.errs.push(ValidationError::new(
                                    "F must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.lhs = built;
                            }
                        }
                        "G" => {
                            read_g = true;
                            if tensor.indices.len() > 1 {
                                env.errs.push(ValidationError::new(
                                    "G must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.rhs = built;
                            }
                        }
                        "out" => {
                            read_out = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                if built.shape.len() > 1 {
                                    env.errs.push(ValidationError::new(
                                        format!("output shape must be a scalar or 1D vector"),
                                        tensor_ast.span,
                                    ));
                                }
                                print!("built out: {:#?}", built);
                                ret.out = built;
                            }
                        }
                        _name => {
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                let env_entry = env.get(built.name).unwrap();
                                let dependent_on_state = env_entry.is_state_dependent();
                                let dependent_on_time = env_entry.is_time_dependent();
                                if !dependent_on_time {
                                    ret.time_indep_defns.push(built);
                                } else if dependent_on_time && !dependent_on_state {
                                    ret.time_dep_defns.push(built);
                                } else {
                                    ret.state_dep_defns.push(built);
                                }
                            }
                        }
                    }
                }
            }
        }
        // set is_algebraic for every state based on env
        for s in &mut ret.states.elmts {
            let env_entry = env.get(s.name).unwrap();
            s.is_algebraic = env_entry.is_algebraic();
        }
        // check that we've read all the required arrays

        let span_all = if ast.is_empty() && ast.first().unwrap().span.is_some() {
            None
        } else {
            Some(StringSpan {
                pos_start: ast.first().unwrap().span.unwrap().pos_start,
                pos_end: ast.last().unwrap().span.unwrap().pos_start,
            })
        };
        if !read_state {
            env.errs.push(ValidationError::new(
                "missing 'u' array".to_string(),
                span_all,
            ));
        }
        if !read_f {
            env.errs.push(ValidationError::new(
                "missing 'F' array".to_string(),
                span_all,
            ));
        }
        if !read_g {
            env.errs.push(ValidationError::new(
                "missing 'G' array".to_string(),
                span_all,
            ));
        }
        if !read_out {
            env.errs.push(ValidationError::new(
                "missing 'out' array".to_string(),
                span_all,
            ));
        }
        // check that length of F and G match the number of states
        if ret.states.shape() != ret.lhs.shape() {
            env.errs.push(ValidationError::new(
                format!(
                    "F and u must have the same shape, but F has shape {} and u has shape {}",
                    ret.lhs.shape(),
                    ret.states.shape()
                ),
                span_all,
            ));
        }
        if ret.states.shape() != ret.rhs.shape() {
            env.errs.push(ValidationError::new(
                format!(
                    "G and u must have the same shape, but G has shape {} and u has shape {}",
                    ret.rhs.shape(),
                    ret.states.shape()
                ),
                span_all,
            ));
        }

        if env.errs.is_empty() {
            Ok(ret)
        } else {
            Err(env.errs)
        }
    }
    pub fn len_state(&self) -> usize {
        self.states.shape()[0]
    }
    pub fn len_inputs(&self) -> usize {
        self.inputs.shape()[0]
    }
    pub fn len_output(&self) -> usize {
        self.out.shape()[0]
    }
    
    fn state_to_elmt(state_cell: &Rc<RefCell<Variable<'s>>>) -> (TensorBlock<'s>, TensorBlock<'s>) {
        let state = state_cell.borrow();
        let ast_eqn = if let Some(eqn) = &state.equation {
            eqn.clone()
        } else {
            panic!("state var should have an equation")
        };
        let (f_astkind, g_astkind) = match ast_eqn.kind {
            AstKind::RateEquation(eqn) => (
                AstKind::new_dot(Ast {
                    kind: AstKind::new_name(state.name),
                    span: ast_eqn.span,
                }),
                eqn.rhs.kind,
            ),
            AstKind::Equation(eqn) => (
                AstKind::new_num(0.0),
                AstKind::new_binop('-', *eqn.rhs, *eqn.lhs),
            ),
            _ => panic!("equation for state var should be rate eqn or standard eqn"),
        };
        (
            TensorBlock {
                start: Index::zeros(1),
                shape: Shape::from_vec(vec![state.dim]),
                expr: Ast {
                    kind: f_astkind,
                    span: ast_eqn.span,
                },
            },
            TensorBlock {
                start: Index::zeros(1),
                shape: Shape::from_vec(vec![state.dim]),
                expr: Ast {
                    kind: g_astkind,
                    span: ast_eqn.span,
                },
            },
        )
    }
    fn state_to_u0(state_cell: &Rc<RefCell<Variable<'s>>>) -> State<'s> {
        let state = state_cell.borrow();
        let init = if state.has_initial_condition() {
            Some(state.init_conditions[0].equation.clone())
        } else {
            None
        };
        State {
            name: state.name,
            shape: Shape::from_vec(vec![state.dim]),
            domain: state.bounds.clone(),
            init,
            start: Index::zeros(1),
            is_algebraic: true,
        }
    }
    fn dfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let defn = defn_cell.borrow();
        Tensor {
            name: defn.name,
            indices: vec!['i'],
            shape: Shape::from_vec(vec![defn.dim]),
            elmts: vec![TensorBlock {
                start: Index::zeros(1),
                shape: Shape::from_vec(vec![defn.dim]),
                expr: defn.expression.as_ref().unwrap().clone(),
            }],
        }
    }

    fn state_to_input(input_cell: &Rc<RefCell<Variable<'s>>>) -> Input<'s> {
        let input = input_cell.borrow();
        assert!(input.is_independent());
        assert!(!input.is_time_dependent());
        Input {
            start: Index::zeros(1),
            name: input.name,
            domain: input.bounds,
            shape: Shape::from_vec(vec![input.dim]),
        }
    }
    fn output_to_elmt(output_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let output = output_cell.borrow();
        TensorBlock {
            expr: Ast {
                kind: AstKind::new_name(output.name),
                span: if output.is_definition() {
                    output.expression.as_ref().unwrap().span
                } else if output.has_equation() {
                    output.equation.as_ref().unwrap().span
                } else {
                    None
                },
            },
            start: Index::zeros(1),
            shape: Shape::from_vec(vec![output.dim]),
        }
    }
    pub fn from(model: &ModelInfo<'s>) -> DiscreteModel<'s> {
        let (time_varying_unknowns, const_unknowns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = model
            .unknowns
            .iter()
            .cloned()
            .partition(|var| var.borrow().is_time_dependent());

        let states: Vec<Rc<RefCell<Variable>>> = time_varying_unknowns
            .iter()
            .filter(|v| v.borrow().is_state())
            .cloned()
            .collect();

        let (state_dep_defns, state_indep_defns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = model
            .definitions
            .iter()
            .cloned()
            .partition(|v| v.borrow().is_dependent_on_state());

        let (time_dep_defns, const_defns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = state_indep_defns
            .iter()
            .cloned()
            .partition(|v| v.borrow().is_time_dependent());

        let mut out_array_elmts: Vec<TensorBlock> =
            chain(time_varying_unknowns.iter(), model.definitions.iter())
                .map(DiscreteModel::output_to_elmt)
                .collect();
        let mut curr_index: usize = 0;
        for elmt in out_array_elmts.iter_mut() {
            elmt.start[0] = i64::try_from(curr_index).unwrap();
            curr_index = curr_index + elmt.shape[0];
        }
        let out_array = Tensor {
            name: "out",
            indices: vec!['i'],
            shape: Shape::from_vec(vec![curr_index]),
            elmts: out_array_elmts,
        };

        let mut f_elmts: Vec<TensorBlock> = Vec::new();
        let mut g_elmts: Vec<TensorBlock> = Vec::new();
        let mut curr_index = 0;
        let mut init_states: Vec<State> = Vec::new();
        for state in states.iter() {
            let mut elmt = DiscreteModel::state_to_elmt(state);
            elmt.0.start[0] = i64::try_from(curr_index).unwrap();
            elmt.1.start[0] = i64::try_from(curr_index).unwrap();
            let mut init_state = DiscreteModel::state_to_u0(state);
            init_state.start[0] = i64::try_from(curr_index).unwrap();
            curr_index = curr_index + elmt.0.shape[0];
            f_elmts.push(elmt.0);
            g_elmts.push(elmt.1);
            init_states.push(init_state);
        }

        let mut curr_index = 0;
        let mut inputs: Vec<Input> = Vec::new();
        for input in const_unknowns.iter() {
            let mut inp = DiscreteModel::state_to_input(input);
            inp.start[0] = i64::try_from(curr_index).unwrap();
            curr_index = curr_index + inp.shape[0];
            inputs.push(inp);
        }

        let state_dep_defns = state_dep_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let time_dep_defns = time_dep_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let time_indep_defns = const_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let lhs_shape = &f_elmts
            .last()
            .unwrap()
            .start
            .mapv(|x| usize::try_from(x).unwrap())
            + &f_elmts.last().unwrap().shape;
        let lhs = Tensor {
            name: "F",
            indices: vec!['i'],
            elmts: f_elmts,
            shape: lhs_shape,
        };
        let rhs_shape = &g_elmts
            .last()
            .unwrap()
            .start
            .mapv(|x| usize::try_from(x).unwrap())
            + &g_elmts.last().unwrap().shape;
        let rhs = Tensor {
            name: "G",
            indices: vec!['i'],
            elmts: g_elmts,
            shape: rhs_shape,
        };
        let name = model.name;
        DiscreteModel {
            name,
            lhs,
            rhs,
            inputs: Inputs::from_vec(inputs),
            states: States::from_vec(init_states, vec!['i']),
            out: out_array,
            time_indep_defns,
            time_dep_defns,
            state_dep_defns,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::ModelInfo, discretise::DiscreteModel, ds_parser, ms_parser::parse_string,
    };

    #[test]
    fn test_circuit_model() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i(t)) {
            let inputVoltage = sin(t) 
            use resistor(v = inputVoltage)
            let doubleI = 2 * i
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.time_indep_defns.len(), 0);
        assert_eq!(discrete.time_dep_defns.len(), 1);
        assert_eq!(discrete.time_dep_defns[0].name, "inputVoltage");
        assert_eq!(discrete.state_dep_defns.len(), 1);
        assert_eq!(discrete.state_dep_defns[0].name, "doubleI");
        assert_eq!(discrete.lhs.name, "F");
        assert_eq!(discrete.rhs.name, "G");
        assert_eq!(discrete.states.shape()[0], 1);
        assert_eq!(discrete.states.elmts().len(), 1);
        assert_eq!(discrete.out.elmts.len(), 3);
        println!("{}", discrete);
    }
    #[test]
    fn rate_equation() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), t, z(t) ) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
            z = 2 * y
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.out.elmts[0].expr.to_string(), "y");
        assert_eq!(discrete.out.elmts[1].expr.to_string(), "t");
        assert_eq!(discrete.out.elmts[2].expr.to_string(), "z");
        println!("{}", discrete);
    }

    #[test]
    fn discrete_logistic_model() {
        const TEXT: &str = "
            in_i {
                r -> [0, inf],
                k -> [0, inf],
            }
            u_i {
                y -> R = 1,
                z -> R,
            }
            F_i {
                dot(y),
                0,
            }
            G_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                y,
                t,
                z,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("logistic_growth", &arrays) {
            Ok(model) => {
                let model_str: String = format!("{}", model).chars().filter(|c| !c.is_whitespace()).collect();
                let text_str: String = TEXT.chars().filter(|c| !c.is_whitespace()).collect();
                assert_eq!(model_str, text_str);
                println!("{}", model);
            }
            Err(e) => {
                panic!("{}", e.as_error_message(TEXT));
            }
        };
    }

    #[test]
    fn discrete_logistic_model_single_state() {
        const TEXT: &str = "
            in {
                r -> [0, inf],
            }
            u {
                y -> R = 1,
            }
            F {
                dot(y),
            }
            G {
                (r * y) * (1 - y),
            }
            out {
                y,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("logistic_growth", &arrays) {
            Ok(model) => {
                let model_str: String = format!("{}", model).chars().filter(|c| !c.is_whitespace()).collect();
                let text_str: String = TEXT.chars().filter(|c| !c.is_whitespace()).collect();
                assert_eq!(model_str, text_str);
                println!("{}", model);
            }
            Err(e) => {
                panic!("{}", e.as_error_message(TEXT));
            }
        };
    }

    #[test]
    fn logistic_model_with_matrix() {
        const TEXT: &str = "
            in_i {
                r -> [0, inf],
                k -> [0, inf],
            }
            I_ij {
                (0, 0): 1,
                (1, 1): 1,
            }
            u_i {
                y -> R^2 = 1,
                z -> R,
            }
            F_i {
                dot(y),
                0,
            }
            rhs_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            G_i {
                sum(j, I_ij * rhs_i)
            }
            out_i {
                y,
                t,
                z,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("logistic_growth", &arrays) {
            Ok(model) => {
                let model_str: String = format!("{}", model).chars().filter(|c| !c.is_whitespace()).collect();
                let text_str: String = TEXT.chars().filter(|c| !c.is_whitespace()).collect();
                assert_eq!(model_str, text_str);
                println!("{}", model);
            }
            Err(e) => {
                panic!("{}", e.as_error_message(TEXT));
            }
        };
    }
 
    
    #[test]
    fn param_error() {
        const TEXT: &str = "
            in_i {
                2 * 1
            }
            u_i {
                y -> R,
            }
            F_i {
                z -> R = 1,
            }
            G {
                y * (1 - (y / k)),
                2 * y
            }
            out_i {
                y,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("test", &arrays) {
            Ok(model) => {
                panic!("Should have failed: {}", model)
            }
            Err(e) => {
                assert!(e.has_error_contains("expected parameter in input"));
                assert!(e.has_error_contains("cannot have parameters in tensor"));
                assert!(e.has_error_contains("cannot have more than one element in a scalar"));
                assert!(e.has_error_contains("cannot find variable k"));
                assert!(e.has_error_contains("F and u must have the same shape"));
                assert!(e.has_error_contains("G and u must have the same shape"));
            }
        };
    }
}
