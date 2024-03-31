use std::boxed::Box;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct DsModel<'a> {
    pub inputs: Vec<&'a str>,
    pub tensors: Vec<Box<Ast<'a>>>,
}

impl<'a> fmt::Display for DsModel<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.inputs.len() > 1 {
            write!(f, "in = [")?;
            for (i, name) in self.inputs.iter().enumerate() {
                write!(f, "{}", name)?;
                if i < self.inputs.len() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
        }
        for tensor in self.tensors.iter() {
            write!(f, "{}", tensor)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Unknown<'a> {
    pub name: &'a str,
    pub dependents: Vec<&'a str>,
    pub codomain: Option<Box<Ast<'a>>>,
}

#[derive(Debug, Clone)]
pub struct Definition<'a> {
    pub name: &'a str,
    pub rhs: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct Submodel<'a> {
    pub name: &'a str,
    pub local_name: &'a str,
    // needs to be a vec of boxes, so that references to a ast node have a consistant type
    pub args: Vec<Box<Ast<'a>>>,
}

#[derive(Debug, Clone)]
pub struct Equation<'a> {
    pub lhs: Box<Ast<'a>>,
    pub rhs: Box<Ast<'a>>,
}

impl<'a> fmt::Display for Equation<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}", self.lhs, self.rhs,)
    }
}

#[derive(Debug, Clone)]
pub struct RateEquation<'a> {
    pub name: &'a str,
    pub rhs: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct IndexedName<'a> {
    pub name: &'a str,
    pub indices: Vec<char>,
}

impl<'a> fmt::Display for IndexedName<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.indices.is_empty() {
            write!(f, "_")?;
            for idx in self.indices.iter() {
                write!(f, "{}", idx)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Range {
    pub lower: f64,
    pub upper: f64,
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.lower, self.upper)
    }
}

#[derive(Debug, Clone)]
pub struct Domain<'a> {
    pub range: Box<Ast<'a>>,
    pub dim: usize,
}

impl<'a> fmt::Display for Domain<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.range).and_then(|_| {
            if self.dim == 1 {
                write!(f, "^{}", self.dim)
            } else {
                Ok(())
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct IntRange {
    pub lower: usize,
    pub upper: usize,
}

#[derive(Debug, Clone)]
pub struct Binop<'a> {
    pub op: char,
    pub left: Box<Ast<'a>>,
    pub right: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct Monop<'a> {
    pub op: char,
    pub child: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct Call<'a> {
    pub fn_name: &'a str,
    pub args: Vec<Box<Ast<'a>>>,
}

#[derive(Debug, Clone)]
pub struct CallArg<'a> {
    pub name: Option<&'a str>,
    pub expression: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct Model<'a> {
    pub name: &'a str,
    pub unknowns: Vec<Box<Ast<'a>>>,
    pub statements: Vec<Box<Ast<'a>>>,
}

#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    name: &'a str,
    indices: Vec<char>,
    elmts: Vec<Ast<'a>>,
}

impl<'a> Tensor<'a> {
    pub fn new(name: &'a str, indices: Vec<char>, elmts: Vec<Ast<'a>>) -> Self {
        Self {
            name,
            indices,
            elmts,
        }
    }

    pub fn elmts(&self) -> &[Ast<'a>] {
        self.elmts.as_ref()
    }

    pub fn name(&self) -> &'a str {
        self.name
    }

    pub fn indices(&self) -> &[char] {
        self.indices.as_ref()
    }
}

impl<'a> fmt::Display for Tensor<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.indices.is_empty() {
            write!(f, "_")?;
            for idx in self.indices.iter() {
                write!(f, "{}", idx)?;
            }
        }
        writeln!(f, " {{")?;
        for elmt in self.elmts.iter() {
            writeln!(f, "{},", elmt)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Indice<'a> {
    pub first: Box<Ast<'a>>,
    pub last: Option<Box<Ast<'a>>>,
    pub sep: Option<&'a str>,
}

impl<'a> fmt::Display for Indice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.first)?;
        if let Some(ref last) = self.last {
            if let Some(ref sep) = self.sep {
                write!(f, "{}{}", sep, last)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Vector<'a> {
    pub data: Vec<Box<Ast<'a>>>,
}

impl<'a> fmt::Display for Vector<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for (i, elmt) in self.data.iter().enumerate() {
            write!(f, "{}", elmt)?;
            if i < self.data.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone)]
pub struct TensorElmt<'a> {
    pub expr: Box<Ast<'a>>,
    pub indices: Option<Box<Ast<'a>>>,
}

impl<'a> TensorElmt<'a> {
    pub fn new(expr: Box<Ast<'a>>, indices: Option<Box<Ast<'a>>>) -> Self {
        Self { expr, indices }
    }
}

#[derive(Debug, Clone)]
pub struct Assignment<'a> {
    pub name: &'a str,
    pub expr: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct Index<'a> {
    pub left: Box<Ast<'a>>,
    pub right: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct Slice<'a> {
    pub lower: Box<Ast<'a>>,
    pub upper: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct NamedGradient<'a> {
    pub gradient_of: Box<Ast<'a>>,
    pub gradient_wrt: Box<Ast<'a>>,
}

impl<'a> fmt::Display for NamedGradient<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "d{}d{}", self.gradient_of, self.gradient_wrt)
    }
}

#[derive(Debug, Clone)]
pub enum AstKind<'a> {
    Model(Model<'a>),
    DsModel(DsModel<'a>),
    Unknown(Unknown<'a>),
    Definition(Definition<'a>),
    Tensor(Tensor<'a>),
    Indice(Indice<'a>),
    Vector(Vector<'a>),
    TensorElmt(TensorElmt<'a>),
    Assignment(Assignment<'a>),
    Submodel(Submodel<'a>),
    Equation(Equation<'a>),
    RateEquation(RateEquation<'a>),
    Range(Range),
    Domain(Domain<'a>),
    IntRange(IntRange),
    Binop(Binop<'a>),
    Monop(Monop<'a>),
    Call(Call<'a>),
    CallArg(CallArg<'a>),
    Index(Index<'a>),
    Slice(Slice<'a>),
    IndexedName(IndexedName<'a>),
    Number(f64),
    Integer(i64),
    Name(&'a str),
    NamedGradient(NamedGradient<'a>),
}

impl<'a> AstKind<'a> {
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            AstKind::Tensor(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_assignment(&self) -> Option<&Assignment> {
        match self {
            AstKind::Assignment(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_model(&self) -> Option<&Model> {
        match self {
            AstKind::Model(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_ds_model(&self) -> Option<&DsModel> {
        match self {
            AstKind::DsModel(m) => Some(m),
            _ => None,
        }
    }
    pub fn to_ds_model(self) -> Option<DsModel<'a>> {
        match self {
            AstKind::DsModel(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_domain(&self) -> Option<&Domain> {
        match self {
            AstKind::Domain(m) => Some(m),
            _ => None,
        }
    }
    pub fn into_model(self) -> Option<Model<'a>> {
        match self {
            AstKind::Model(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_call_arg(&self) -> Option<&CallArg> {
        match self {
            AstKind::CallArg(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_named_gradient(&self) -> Option<&NamedGradient> {
        match self {
            AstKind::NamedGradient(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_name(&self) -> Option<&str> {
        match self {
            AstKind::Name(n) => Some(n),
            AstKind::IndexedName(n) => Some(n.name),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&Tensor> {
        match self {
            AstKind::Tensor(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_vector(&self) -> Option<&Vector> {
        match self {
            AstKind::Vector(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_real(&self) -> Option<f64> {
        match self {
            AstKind::Number(a) => Some(*a),
            AstKind::Integer(a) => Some(*a as f64),
            _ => None,
        }
    }
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            AstKind::Integer(a) => Some(*a),
            AstKind::Number(a) => Some(*a as i64),
            _ => None,
        }
    }
    pub fn as_indice(&self) -> Option<&Indice> {
        match self {
            AstKind::Indice(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_tensor_elmt(&self) -> Option<&TensorElmt> {
        match self {
            AstKind::TensorElmt(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_range(&self) -> Option<&Range> {
        match self {
            AstKind::Range(a) => Some(a),
            _ => None,
        }
    }
    pub fn into_array(self) -> Option<Tensor<'a>> {
        match self {
            AstKind::Tensor(a) => Some(a),
            _ => None,
        }
    }
    pub fn new_binop(op: char, left: Ast<'a>, right: Ast<'a>) -> Self {
        AstKind::Binop(Binop {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    }
    pub fn new_dot(child: Ast<'a>) -> Self {
        AstKind::Call(Call {
            fn_name: "dot",
            args: vec![Box::new(child)],
        })
    }
    pub fn new_index(left: Ast<'a>, right: Ast<'a>) -> Self {
        AstKind::Index(Index {
            left: Box::new(left),
            right: Box::new(right),
        })
    }
    pub fn new_name(name: &'a str) -> Self {
        AstKind::Name(name)
    }
    pub fn new_indexed_name(name: &'a str, indices: Vec<char>) -> Self {
        AstKind::IndexedName(IndexedName { name, indices })
    }
    pub fn new_time_derivative(name: &'a str) -> Self {
        AstKind::NamedGradient(NamedGradient {
            gradient_of: Box::new(Ast {
                kind: Self::new_name(name),
                span: None,
            }),
            gradient_wrt: Box::new(Ast {
                kind: Self::new_name("t"),
                span: None,
            }),
        })
    }
    pub fn new_int(num: i64) -> Self {
        AstKind::Integer(num)
    }
    pub fn new_irange(range: (usize, usize)) -> Self {
        AstKind::IntRange(IntRange {
            lower: range.0,
            upper: range.1,
        })
    }
    pub fn new_num(num: f64) -> Self {
        AstKind::Number(num)
    }
    pub fn new_tensor(name: &'a str, indices: Vec<char>, elmts: Vec<Ast<'a>>) -> Self {
        AstKind::Tensor(Tensor::new(name, indices, elmts))
    }
    pub fn new_tensor_elmt(expr: Ast<'a>, indices: Option<Ast<'a>>) -> Self {
        AstKind::TensorElmt(TensorElmt::new(Box::new(expr), indices.map(Box::new)))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct StringSpan {
    pub pos_start: usize,
    pub pos_end: usize,
}

impl Add for StringSpan {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            pos_start: cmp::min(self.pos_start, other.pos_start),
            pos_end: cmp::max(self.pos_end, other.pos_end),
        }
    }
}

impl fmt::Display for StringSpan {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.pos_start, self.pos_end)
    }
}

#[derive(Debug, Clone)]
pub struct Ast<'a> {
    pub kind: AstKind<'a>,
    pub span: Option<StringSpan>,
}

impl<'a> Ast<'a> {
    pub fn new_binop(op: char, lhs: Ast<'a>, rhs: Ast<'a>) -> Self {
        Ast {
            kind: AstKind::new_binop(op, lhs, rhs),
            span: None,
        }
    }

    pub fn clone_and_subst<'b>(&self, replacements: &HashMap<&'a str, &'b Ast<'a>>) -> Self {
        let cloned_kind = match &self.kind {
            AstKind::Definition(dfn) => AstKind::Definition(Definition {
                name: dfn.name,
                rhs: Box::new(dfn.rhs.clone_and_subst(replacements)),
            }),
            AstKind::Equation(eqn) => AstKind::Equation(Equation {
                lhs: Box::new(eqn.lhs.clone_and_subst(replacements)),
                rhs: Box::new(eqn.rhs.clone_and_subst(replacements)),
            }),
            AstKind::RateEquation(eqn) => AstKind::RateEquation(RateEquation {
                name: eqn.name,
                rhs: Box::new(eqn.rhs.clone_and_subst(replacements)),
            }),
            AstKind::Binop(binop) => AstKind::Binop(Binop {
                op: binop.op,
                left: Box::new(binop.left.clone_and_subst(replacements)),
                right: Box::new(binop.right.clone_and_subst(replacements)),
            }),
            AstKind::Index(binop) => AstKind::Index(Index {
                left: Box::new(binop.left.clone_and_subst(replacements)),
                right: Box::new(binop.right.clone_and_subst(replacements)),
            }),
            AstKind::Slice(slice) => AstKind::Slice(Slice {
                lower: Box::new(slice.lower.clone_and_subst(replacements)),
                upper: Box::new(slice.upper.clone_and_subst(replacements)),
            }),
            AstKind::Monop(binop) => AstKind::Monop(Monop {
                op: binop.op,
                child: Box::new(binop.child.clone_and_subst(replacements)),
            }),
            AstKind::Call(call) => AstKind::Call(Call {
                fn_name: call.fn_name,
                args: call
                    .args
                    .iter()
                    .map(|m| Box::new(m.clone_and_subst(replacements)))
                    .collect(),
            }),
            AstKind::CallArg(arg) => AstKind::CallArg(CallArg {
                name: arg.name,
                expression: Box::new(arg.expression.clone_and_subst(replacements)),
            }),
            AstKind::Number(num) => AstKind::Number(*num),
            AstKind::Integer(num) => AstKind::Integer(*num),
            AstKind::IndexedName(name) => AstKind::IndexedName(IndexedName {
                name: name.name,
                indices: name.indices.clone(),
            }),
            AstKind::Name(name) => {
                if let Some(x) = replacements.get(name) {
                    x.kind.clone()
                } else {
                    AstKind::Name(name)
                }
            }
            AstKind::NamedGradient(gradient) => AstKind::NamedGradient(NamedGradient {
                gradient_of: Box::new(gradient.gradient_of.clone_and_subst(replacements)),
                gradient_wrt: Box::new(gradient.gradient_wrt.clone_and_subst(replacements)),
            }),
            AstKind::Model(m) => AstKind::Model(m.clone()),
            AstKind::Unknown(unknown) => AstKind::Unknown(unknown.clone()),
            AstKind::Submodel(submodel) => AstKind::Submodel(Submodel {
                name: submodel.name,
                local_name: submodel.local_name,
                args: submodel
                    .args
                    .iter()
                    .map(|m| Box::new(m.clone_and_subst(replacements)))
                    .collect(),
            }),
            AstKind::Range(range) => AstKind::Range(range.clone()),
            AstKind::Domain(domain) => AstKind::Domain(domain.clone()),
            AstKind::Indice(indices) => AstKind::Indice(indices.clone()),
            AstKind::IntRange(range) => AstKind::IntRange(range.clone()),
            AstKind::DsModel(m) => AstKind::DsModel(m.clone()),
            AstKind::Tensor(a) => AstKind::Tensor(a.clone()),
            AstKind::TensorElmt(a) => AstKind::TensorElmt(a.clone()),
            AstKind::Assignment(a) => AstKind::Assignment(a.clone()),
            AstKind::Vector(a) => AstKind::Vector(a.clone()),
        };
        Ast {
            kind: cloned_kind,
            span: self.span,
        }
    }

    pub fn get_dependents(&self) -> HashSet<&'a str> {
        let mut deps = HashSet::new();
        self.collect_deps(&mut deps);
        deps
    }

    fn collect_deps(&self, deps: &mut HashSet<&'a str>) {
        match &self.kind {
            AstKind::Equation(eqn) => {
                eqn.lhs.collect_deps(deps);
                eqn.rhs.collect_deps(deps);
            }
            AstKind::RateEquation(eqn) => {
                deps.insert(eqn.name);
                eqn.rhs.collect_deps(deps);
            }
            AstKind::Binop(binop) => {
                binop.left.collect_deps(deps);
                binop.right.collect_deps(deps);
            }
            AstKind::Monop(monop) => {
                monop.child.collect_deps(deps);
            }
            AstKind::Call(call) => {
                let mut arg = call.args.iter();
                // don't count the first argument of sum as a dependency
                if call.fn_name == "sum" {
                    arg.next();
                }
                for c in arg {
                    c.collect_deps(deps);
                }
            }
            AstKind::CallArg(arg) => {
                arg.expression.collect_deps(deps);
            }
            AstKind::IndexedName(found_name) => {
                deps.insert(found_name.name);
            }
            AstKind::Name(found_name) => {
                deps.insert(found_name);
            }
            AstKind::NamedGradient(gradient) => {
                gradient.gradient_of.collect_deps(deps);
            }
            AstKind::Index(index) => {
                index.left.collect_deps(deps);
                index.right.collect_deps(deps);
            }
            AstKind::Slice(slice) => {
                slice.lower.collect_deps(deps);
                slice.upper.collect_deps(deps);
            }
            AstKind::Tensor(tensor) => {
                for elmt in &tensor.elmts {
                    elmt.collect_deps(deps);
                }
            }
            AstKind::TensorElmt(elmt) => {
                elmt.expr.collect_deps(deps);
            }
            AstKind::DsModel(m) => deps.extend(m.inputs.iter().cloned()),
            AstKind::Number(_) => (),
            AstKind::Integer(_) => (),
            AstKind::Model(_) => (),
            AstKind::Unknown(_) => (),
            AstKind::Definition(_) => (),
            AstKind::Submodel(_) => (),
            AstKind::Range(_) => (),
            AstKind::Domain(_) => (),
            AstKind::IntRange(_) => (),
            AstKind::Assignment(_) => (),
            AstKind::Vector(_) => (),
            AstKind::Indice(_) => (),
        }
    }

    pub fn get_indices(&self) -> Vec<char> {
        let mut indices = Vec::new();
        self.collect_indices(&mut indices);
        indices
    }

    fn collect_indices(&self, indices: &mut Vec<char>) {
        match &self.kind {
            AstKind::Assignment(a) => {
                a.expr.collect_indices(indices);
            }
            AstKind::Equation(eqn) => {
                eqn.lhs.collect_indices(indices);
                eqn.rhs.collect_indices(indices);
            }
            AstKind::RateEquation(eqn) => {
                eqn.rhs.collect_indices(indices);
            }
            AstKind::Binop(binop) => {
                binop.left.collect_indices(indices);
                binop.right.collect_indices(indices);
            }
            AstKind::Monop(monop) => {
                monop.child.collect_indices(indices);
            }
            AstKind::Call(call) => {
                for c in &call.args {
                    c.collect_indices(indices);
                }
            }
            AstKind::CallArg(arg) => {
                arg.expression.collect_indices(indices);
            }
            AstKind::IndexedName(found_name) => {
                indices.extend(found_name.indices.iter().cloned());
            }
            AstKind::Index(index) => {
                index.left.collect_indices(indices);
                index.right.collect_indices(indices);
            }
            AstKind::Slice(slice) => {
                slice.lower.collect_indices(indices);
                slice.upper.collect_indices(indices);
            }
            AstKind::Tensor(tensor) => {
                for elmt in &tensor.elmts {
                    elmt.collect_indices(indices);
                }
            }
            AstKind::TensorElmt(elmt) => {
                elmt.expr.collect_indices(indices);
            }
            AstKind::NamedGradient(gradient) => {
                gradient.gradient_of.collect_indices(indices);
                gradient.gradient_wrt.collect_indices(indices);
            }
            AstKind::Name(_) => (),
            AstKind::DsModel(_) => (),
            AstKind::Number(_) => (),
            AstKind::Integer(_) => (),
            AstKind::Model(_) => (),
            AstKind::Unknown(_) => (),
            AstKind::Definition(_) => (),
            AstKind::Submodel(_) => (),
            AstKind::Range(_) => (),
            AstKind::Domain(_) => (),
            AstKind::IntRange(_) => (),
            AstKind::Vector(_) => (),
            AstKind::Indice(_) => (),
        }
    }
}

impl<'a> fmt::Display for Ast<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.kind {
            AstKind::Model(model) => {
                write!(
                    f,
                    "Model {} {:#?} {:#?}",
                    model.name, model.unknowns, model.statements
                )
            }
            AstKind::Name(name) => write!(f, "{}", name),
            AstKind::IndexedName(name) => write!(f, "{}", name),
            AstKind::Number(num) => write!(f, "{}", num),
            AstKind::Integer(num) => write!(f, "{}", num),
            AstKind::Unknown(unknown) => write!(
                f,
                "Unknown ({})({:#?}) -> {:#?}",
                unknown.name, unknown.dependents, unknown.codomain
            ),
            AstKind::Domain(domain) => write!(f, "{}", domain.range).and_then(|_| {
                if domain.dim == 1 {
                    write!(f, "^{}", domain.dim)
                } else {
                    Ok(())
                }
            }),
            AstKind::IntRange(range) => write!(f, "({}, {})", range.lower, range.upper),
            AstKind::Equation(eqn) => {
                write!(f, "{}", eqn)
            }
            AstKind::RateEquation(reqn) => {
                write!(f, "dot({}) = {}", reqn.name, reqn.rhs)
            }
            AstKind::Submodel(submodel) => write!(
                f,
                "Submodel {} {:#?} as {}",
                submodel.name, submodel.args, submodel.local_name
            ),
            AstKind::Binop(binop) => {
                let lhs_bracket = matches!(binop.left.kind, AstKind::Binop(_) | AstKind::Monop(_));
                let rhs_bracket = matches!(binop.right.kind, AstKind::Binop(_) | AstKind::Monop(_));
                let lhs = if lhs_bracket {
                    format!("({})", binop.left)
                } else {
                    format!("{}", binop.left)
                };
                let rhs = if rhs_bracket {
                    format!("({})", binop.right)
                } else {
                    format!("{}", binop.right)
                };
                write!(f, "{} {} {}", lhs, binop.op, rhs,)
            }
            AstKind::Monop(monop) => {
                let bracket = matches!(monop.child.kind, AstKind::Binop(_) | AstKind::Monop(_));
                if bracket {
                    write!(f, "{} ({})", monop.op, monop.child)
                } else {
                    write!(f, "{} {}", monop.op, monop.child)
                }
            }
            AstKind::Call(call) => {
                let arg_strs: Vec<String> = call.args.iter().map(|arg| arg.to_string()).collect();
                write!(f, "{}({})", call.fn_name, arg_strs.join(", "))
            }
            AstKind::CallArg(arg) => match arg.name {
                Some(name) => write!(f, "{} = {}", name, arg.expression),
                None => write!(f, "{}", arg.expression),
            },
            AstKind::Definition(dfn) => {
                write!(f, "{} = {}", dfn.name, dfn.rhs,)
            }
            AstKind::Index(index) => {
                write!(f, "{}[{}]", index.left, index.right)
            }
            AstKind::Slice(slice) => {
                write!(f, "{}:{}", slice.lower, slice.upper)
            }
            AstKind::TensorElmt(elmt) => {
                if let Some(indices) = &elmt.indices {
                    write!(f, "{} {}", indices, elmt.expr)
                } else {
                    write!(f, "{}", elmt.expr)
                }
            }
            AstKind::Assignment(a) => {
                write!(f, "{} = {}", a.name, a.expr)
            }
            AstKind::DsModel(m) => write!(f, "{}", m),
            AstKind::Tensor(tensor) => write!(f, "{}", tensor),
            AstKind::Range(range) => write!(f, "{}", range),
            AstKind::Vector(v) => write!(f, "{}", v),
            AstKind::Indice(i) => write!(f, "{}", i),
            AstKind::NamedGradient(gradient) => write!(f, "{}", gradient),
        }
    }
}
