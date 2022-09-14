use std::fmt;
use std::rc::Rc;
use crate::semantic::ModelInfo;

#[derive(Debug)]
pub enum AstKind<'a> {
    Model {
        name: &'a str,
        unknowns: Vec<Rc<Ast<'a>>>,
        statements: Vec<Rc<Ast<'a>>>,
        info: Option<ModelInfo<'a>>,
    },
    Unknown {
        name: &'a str,
        dependents: Vec<&'a str>,
        codomain: Option<Rc<Ast<'a>>>,
    },

    Definition {
        name: &'a str,
        rhs: Rc<Ast<'a>>,
    },

    Submodel {
        name: &'a str,
        local_name: &'a str,
        args: Vec<Ast<'a>>,
    },

    Equation {
        lhs: Rc<Ast<'a>>,
        rhs: Rc<Ast<'a>>,
    },

    RateEquation {
        name: &'a str,
        rhs: Rc<Ast<'a>>,
    },

    Range {
        lower: f64,
        upper: f64,
    },

    Binop {
        op: char,
        left: Rc<Ast<'a>>,
        right: Rc<Ast<'a>>,
    },

    Monop {
        op: char,
        child: Rc<Ast<'a>>,
    },

    Call {
        fn_name: &'a str,
        args: Vec<Ast<'a>>,
    },

    CallArg {
        name: Option<&'a str>,
        expression: Rc<Ast<'a>>,
    },

    Number(f64),

    Name(&'a str),
}


#[derive(Debug)]
pub struct Ast<'a> {
    pub kind: AstKind<'a>,
    pub pos_start: usize,
    pub pos_end: usize,
}

fn expr_to_string(ast: &Ast) -> String {
    match ast.kind {
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
        match self.kind {
            AstKind::Model {
                name,
                unknowns,
                statements,
                info,
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

