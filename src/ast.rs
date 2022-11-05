use std::boxed::Box;
use std::collections::HashMap;
use std::fmt;
use std::mem::replace;

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

#[derive(Debug, Clone)]
pub struct RateEquation<'a> {
    pub name: &'a str,
    pub rhs: Box<Ast<'a>>,
}

#[derive(Debug, Clone)]
pub struct Range {
    pub lower: f64,
    pub upper: f64,
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
pub enum AstKind<'a> {
    Model(Model<'a>),
    Unknown(Unknown<'a>),
    Definition(Definition<'a>),
    Submodel(Submodel<'a>),
    Equation(Equation<'a>),
    RateEquation(RateEquation<'a>),
    Range(Range),
    Binop(Binop<'a>),
    Monop(Monop<'a>),
    Call(Call<'a>),
    CallArg(CallArg<'a>),
    Number(f64),
    Name(&'a str),
}

impl<'a> AstKind<'a> {
    pub fn clone_and_subst(
        self: Self,
        replacements: HashMap<&'a str, Box<Ast<'a>>>,
    ) -> (Self, Option<Box<Ast<'a>>>) {
        match self {
            AstKind::Definition(dfn) => (
                AstKind::Definition(Definition {
                    name: dfn.name.clone(),
                    rhs: Box::new(dfn.rhs.clone_and_subst(replacements)),
                }),
                None,
            ),
            AstKind::Equation(eqn) => (
                AstKind::Equation(Equation {
                    lhs: Box::new(eqn.lhs.clone_and_subst(replacements)),
                    rhs: Box::new(eqn.rhs.clone_and_subst(replacements)),
                }),
                None,
            ),
            AstKind::RateEquation(eqn) => (
                AstKind::RateEquation(RateEquation {
                    name: eqn.name.clone(),
                    rhs: Box::new(eqn.rhs.clone_and_subst(replacements)),
                }),
                None,
            ),
            AstKind::Binop(binop) => (
                AstKind::Binop(Binop {
                    op: binop.op.clone(),
                    left: Box::new(binop.left.clone_and_subst(replacements)),
                    right: Box::new(binop.left.clone_and_subst(replacements)),
                }),
                None,
            ),
            AstKind::Monop(binop) => (
                AstKind::Monop(Monop {
                    op: binop.op.clone(),
                    child: Box::new(binop.child.clone_and_subst(replacements)),
                }),
                None,
            ),
            AstKind::Call(call) => (
                AstKind::Call(Call {
                    fn_name: call.fn_name.clone(),
                    args: call
                        .args
                        .into_iter()
                        .map(|m| m.clone_and_subst(replacements))
                        .collect(),
                }),
                None,
            ),
            AstKind::CallArg(arg) => (
                AstKind::CallArg(CallArg {
                    name: arg.name.clone(),
                    expression: Box::new(arg.expression.clone_and_subst(replacements)),
                }),
                None,
            ),
            AstKind::Number(num) => (AstKind::Number(num), None),
            AstKind::Name(name) => match replacements.get(name) {
                Some(&x) => (x.kind, Some(x)),
                None => (AstKind::Name(name), None),
            },
            AstKind::Model(m) => (
                AstKind::Model(m.clone()),
                None,
            ),
            AstKind::Unknown(unknown) => (
                AstKind::Unknown(unknown.clone()),
                None,
            )
            AstKind::Submodel(submodel) => (
                AstKind::Submodel(Submodel{ 
                    name: submodel.name.clone(), 
                    local_name: submodel.local_name.clone(), 
                    args: submodel
                        .args
                        .into_iter()
                        .map(|m| m.clone_and_subst(replacements))
                        .collect(),
                }),
                None,
            ),
            AstKind::Range(range) => (
                AstKind::Range(range.clone()),
                None,
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StringSpan {
    pub pos_start: usize,
    pub pos_end: usize,
}

#[derive(Debug, Clone)]
pub struct Ast<'a> {
    pub kind: AstKind<'a>,
    pub span: StringSpan,
}

impl<'a> Ast<'a> {
    pub fn clone_and_subst(&self, replacements: HashMap<&'a str, Box<Ast<'a>>>) -> Self {
        let (cloned_kind, repl) = self.kind.clone_and_subst(replacements);
        Ast {
            kind: cloned_kind,
            span: self.span.clone(),
        }
    }
}

fn expr_to_string(ast: &Ast) -> String {
    match &ast.kind {
        AstKind::Binop(binop) => {
            format!(
                "{} {} {}",
                expr_to_string(binop.left.as_ref()),
                binop.op,
                expr_to_string(binop.right.as_ref())
            )
        }
        AstKind::Name(value) => value.to_string(),
        AstKind::Monop(monop) => format!("{} {}", monop.op, expr_to_string(monop.child.as_ref())),
        AstKind::Number(value) => value.to_string(),
        _ => unreachable!(),
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
            AstKind::Name(name) => write!(f, "Name({})", name),
            AstKind::Number(num) => write!(f, "Number({})", num),
            AstKind::Unknown(unknown) => write!(
                f,
                "Unknown ({})({:#?}) -> {:#?}",
                unknown.name, unknown.dependents, unknown.codomain
            ),
            AstKind::Range(range) => write!(f, "({}, {})", range.lower, range.upper),
            AstKind::Equation(eqn) => {
                write!(
                    f,
                    "{} = {}",
                    expr_to_string(eqn.lhs.as_ref()),
                    expr_to_string(eqn.rhs.as_ref())
                )
            }
            AstKind::RateEquation(reqn) => {
                write!(
                    f,
                    "dot({}) = {}",
                    reqn.name,
                    expr_to_string(reqn.rhs.as_ref())
                )
            }
            AstKind::Submodel(submodel) => write!(
                f,
                "Submodel {} {:#?} as {}",
                submodel.name, submodel.args, submodel.local_name
            ),
            _ => unreachable!(),
        }
    }
}
