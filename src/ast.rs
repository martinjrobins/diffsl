use std::fmt;

#[derive(Debug)]
pub enum Ast {
    Model {
        name: String,
        unknowns: Vec<Box<Ast>>,
        statements: Vec<Box<Ast>>,
    },
    Unknown {
        name: String,
        dependents: Vec<String>,
        codomain: Option<Box<Ast>>,
    },

    Definition {
        name: String,
        rhs: Box<Ast>,
    },

    Submodel {
        name: String,
        local_name: String,
        args: Vec<Ast>,
    },

    Equation {
        lhs: Box<Ast>,
        rhs: Box<Ast>,
    },

    RateEquation {
        name: String,
        rhs: Box<Ast>,
    },

    Range {
        lower: f64,
        upper: f64,
    },

    Binop {
        op: char,
        left: Box<Ast>,
        right: Box<Ast>,
    },

    Monop {
        op: char,
        child: Box<Ast>,
    },

    Call {
        fn_name: String,
        args: Vec<Ast>,
    },

    CallArg {
        name: Option<String>,
        expression: Box<Ast>,
    },

    Number(f64),

    Name(String),
}

pub fn expr_to_string(ast: &Ast) -> String {
    match ast {
        Ast::Binop { op, left, right } => {
            format!("{} {} {}", expr_to_string(left), op, expr_to_string(right))
        }
        Ast::Monop { op, child } => format!("{} {}", op, expr_to_string(child)),
        Ast::Name(value) => value.to_string(),
        Ast::Number(value) => value.to_string(),
        _ => unreachable!(),
    }
}

impl fmt::Display for Ast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ast::Model {
                name,
                unknowns,
                statements,
            } => {
                write!(f, "Model {} {:#?} {:#?}", name, unknowns, statements)
            }
            Ast::Name(name) => write!(f, "Name({})", name),
            Ast::Number(num) => write!(f, "Number({})", num),
            Ast::Unknown {
                name,
                dependents,
                codomain,
            } => write!(
                f,
                "Unknown ({})({:#?}) -> {:#?}",
                name, dependents, codomain
            ),
            Ast::Range { lower, upper } => write!(f, "({}, {})", lower, upper),
            Ast::Equation { lhs, rhs } => {
                write!(f, "{} = {}", expr_to_string(lhs), expr_to_string(rhs))
            }
            Ast::RateEquation { name, rhs } => write!(f, "dot({}) = {}", name, expr_to_string(rhs)),
            Ast::Submodel {
                name,
                local_name,
                args,
            } => write!(f, "Submodel {} {:#?} as {}", name, args, local_name),
            _ => unreachable!(),
        }
    }
}

