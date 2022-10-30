use std::boxed::Box;
use std::fmt;

#[derive(Debug)]
pub struct Unknown<'a> {
    pub name: &'a str,
    pub dependents: Vec<&'a str>,
    pub codomain: Option<Box<Ast<'a>>>,
}

#[derive(Debug)]
pub struct Definition<'a> {
    pub name: &'a str,
    pub rhs: Box<Ast<'a>>,
}

#[derive(Debug)]
pub struct Submodel<'a> {
    pub name: &'a str,
    pub local_name: &'a str,
    pub args: Vec<Box<Ast<'a>>>,
}

#[derive(Debug)]
pub struct Equation<'a> {
    pub lhs: Box<Ast<'a>>,
    pub rhs: Box<Ast<'a>>,
}

#[derive(Debug)]
pub struct RateEquation<'a> {
    pub name: &'a str,
    pub rhs: Box<Ast<'a>>,
}

#[derive(Debug)]
pub struct Range {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug)]
pub struct Binop<'a> {
    pub op: char,
    pub left: Box<Ast<'a>>,
    pub right: Box<Ast<'a>>,
}

#[derive(Debug)]
pub struct Monop<'a> {
    pub op: char,
    pub child: Box<Ast<'a>>,
}

#[derive(Debug)]
pub struct Call<'a> {
    pub fn_name: &'a str,
    pub args: Vec<Box<Ast<'a>>>,
}

#[derive(Debug)]
pub struct CallArg<'a> {
    pub name: Option<&'a str>,
    pub expression: Box<Ast<'a>>,
}

#[derive(Debug)]
pub struct Model<'a> {
    pub name: &'a str,
    pub unknowns: Vec<Box<Ast<'a>>>,
    pub statements: Vec<Box<Ast<'a>>>,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct StringSpan {
    pub pos_start: usize,
    pub pos_end: usize,
}

#[derive(Debug)]
pub struct Ast<'a> {
    pub kind: AstKind<'a>,
    pub span: StringSpan,
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
                write!(f, "Model {} {:#?} {:#?}", model.name, model.unknowns, model.statements)
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
                write!(f, "dot({}) = {}", reqn.name, expr_to_string(reqn.rhs.as_ref()))
            }
            AstKind::Submodel(submodel) => write!(f, "Submodel {} {:#?} as {}", submodel.name, submodel.args, submodel.local_name),
            _ => unreachable!(),
        }
    }
}
