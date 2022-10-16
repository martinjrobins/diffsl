use std::fmt;
use std::boxed::Box;

#[derive(Debug)]
pub enum AstKind<'a> {
    Model {
        name: &'a str,
        unknowns: Vec<Box<Ast<'a>>>,
        statements: Vec<Box<Ast<'a>>>,
    },
    Unknown {
        name: &'a str,
        dependents: Vec<&'a str>,
        codomain: Option<Box<Ast<'a>>>,
    },

    Definition {
        name: &'a str,
        rhs: Box<Ast<'a>>,
    },

    Submodel {
        name: &'a str,
        local_name: &'a str,
        args: Vec<Ast<'a>>,
    },

    Equation {
        lhs: Box<Ast<'a>>,
        rhs: Box<Ast<'a>>,
    },

    RateEquation {
        name: &'a str,
        rhs: Box<Ast<'a>>,
    },

    Range {
        lower: f64,
        upper: f64,
    },

    Binop {
        op: char,
        left: Box<Ast<'a>>,
        right: Box<Ast<'a>>,
    },

    Monop {
        op: char,
        child: Box<Ast<'a>>,
    },

    Call {
        fn_name: &'a str,
        args: Vec<Ast<'a>>,
    },

    CallArg {
        name: Option<&'a str>,
        expression: Box<Ast<'a>>,
    },

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
        AstKind::Binop { op, left, right } => {
            format!("{} {} {}", expr_to_string(left.as_ref()), op, expr_to_string(right.as_ref()))
        }
        AstKind::Name(value) => value.to_string(),
        AstKind::Monop { op, child } => format!("{} {}", op, expr_to_string(child.as_ref())),
        AstKind::Number(value) => value.to_string(),
        _ => unreachable!(),
    }
}

impl<'a> fmt::Display for Ast<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.kind {
            AstKind::Model {
                name,
                unknowns,
                statements,
            } => {
                write!(f, "Model {} {:#?} {:#?}", name, unknowns, statements)
            }
            AstKind::Name(name) => write!(f, "Name({})", name),
            AstKind::Number(num) => write!(f, "Number({})", num),
            AstKind::Unknown {
                name,
                dependents,
                codomain,
            } => write!(
                f,
                "Unknown ({})({:#?}) -> {:#?}",
                name, dependents, codomain
            ),
            AstKind::Range { lower, upper } => write!(f, "({}, {})", lower, upper),
            AstKind::Equation { lhs, rhs } => {
                write!(f, "{} = {}", expr_to_string(lhs.as_ref()), expr_to_string(rhs.as_ref()))
            }
            AstKind::RateEquation { name, rhs } => write!(f, "dot({}) = {}", name, expr_to_string(rhs.as_ref())),
            AstKind::Submodel {
                name,
                local_name,
                args,
            } => write!(f, "Submodel {} {:#?} as {}", name, args, local_name),
            _ => unreachable!(),
        }
    }
}

