use std::boxed::Box;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::ops::Add;

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
    pub initial_condition_for: Option<&'a str>,
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
pub struct Index<'a> {
    pub name: &'a str,
    pub index: usize,
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
    Index(Index<'a>),
    Call(Call<'a>),
    CallArg(CallArg<'a>),
    Number(f64),
    Name(&'a str),
}

impl<'a> AstKind<'a> {
    pub fn model(&self) -> Option<&Model> {
        match self {
            AstKind::Model(m) => Some(m),
            _ => None,
        }
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

#[derive(Debug, Clone)]
pub struct Ast<'a> {
    pub kind: AstKind<'a>,
    pub span: StringSpan,
}

impl<'a> Ast<'a> {
    pub fn clone_and_subst<'b>(&self, replacements: &HashMap<&'a str, &'b Box<Ast<'a>>>) -> Self {
        let cloned_kind = match &self.kind {
            AstKind::Definition(dfn) => AstKind::Definition(Definition {
                name: dfn.name.clone(),
                rhs: Box::new(dfn.rhs.clone_and_subst(replacements)),
            }),
            AstKind::Equation(eqn) => AstKind::Equation(Equation {
                lhs: Box::new(eqn.lhs.clone_and_subst(replacements)),
                rhs: Box::new(eqn.rhs.clone_and_subst(replacements)),
                initial_condition_for: eqn.initial_condition_for.clone(),
            }),
            AstKind::RateEquation(eqn) => AstKind::RateEquation(RateEquation {
                name: eqn.name.clone(),
                rhs: Box::new(eqn.rhs.clone_and_subst(replacements)),
            }),
            AstKind::Binop(binop) => AstKind::Binop(Binop {
                op: binop.op.clone(),
                left: Box::new(binop.left.clone_and_subst(replacements)),
                right: Box::new(binop.right.clone_and_subst(replacements)),
            }),
            AstKind::Monop(binop) => AstKind::Monop(Monop {
                op: binop.op.clone(),
                child: Box::new(binop.child.clone_and_subst(replacements)),
            }),
            AstKind::Call(call) => AstKind::Call(Call {
                fn_name: call.fn_name.clone(),
                args: call
                    .args
                    .iter()
                    .map(|m| Box::new(m.clone_and_subst(replacements)))
                    .collect(),
            }),
            AstKind::CallArg(arg) => AstKind::CallArg(CallArg {
                name: arg.name.clone(),
                expression: Box::new(arg.expression.clone_and_subst(replacements)),
            }),
            AstKind::Number(num) => AstKind::Number(*num),
            AstKind::Name(name) => {
                if let Some(x) = replacements.get(name) {
                    x.kind.clone()
                } else {
                    AstKind::Name(name)
                }
            }
            AstKind::Model(m) => AstKind::Model(m.clone()),
            AstKind::Unknown(unknown) => AstKind::Unknown(unknown.clone()),
            AstKind::Submodel(submodel) => AstKind::Submodel(Submodel {
                name: submodel.name.clone(),
                local_name: submodel.local_name.clone(),
                args: submodel
                    .args
                    .iter()
                    .map(|m| Box::new(m.clone_and_subst(replacements)))
                    .collect(),
            }),
            AstKind::Range(range) => AstKind::Range(range.clone()),
            AstKind::Index(index) => AstKind::Index(index.clone()),
        };
        Ast {
            kind: cloned_kind,
            span: self.span.clone(),
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
                for c in call.args {
                    c.collect_deps(deps);
                }
            }
            AstKind::CallArg(arg) => {
                arg.expression.collect_deps(deps);
            }
            AstKind::Name(found_name) => {
                deps.insert(found_name);
            }
            AstKind::Index(index) => {
                deps.insert(index.name);
            },
            AstKind::Number(_) => (),
            AstKind::Model(_) => (),
            AstKind::Unknown(_) => (),
            AstKind::Definition(_) => (),
            AstKind::Submodel(_) => (),
            AstKind::Range(_) => (),
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
            AstKind::Number(num) => write!(f, "{}", num),
            AstKind::Unknown(unknown) => write!(
                f,
                "Unknown ({})({:#?}) -> {:#?}",
                unknown.name, unknown.dependents, unknown.codomain
            ),
            AstKind::Range(range) => write!(f, "({}, {})", range.lower, range.upper),
            AstKind::Equation(eqn) => {
                write!(f, "{} = {}", eqn.lhs.to_string(), eqn.rhs.to_string(),)
            }
            AstKind::RateEquation(reqn) => {
                write!(f, "dot({}) = {}", reqn.name, reqn.rhs.to_string())
            }
            AstKind::Submodel(submodel) => write!(
                f,
                "Submodel {} {:#?} as {}",
                submodel.name, submodel.args, submodel.local_name
            ),
            AstKind::Binop(binop) => {
                write!(
                    f,
                    "{} {} {}",
                    binop.left.to_string(),
                    binop.op,
                    binop.right.to_string(),
                )
            }
            AstKind::Monop(monop) => write!(f, "{} {}", monop.op, monop.child.to_string()),
            AstKind::Call(call) => {
                let arg_strs: Vec<String> = call.args.iter().map(|arg| arg.to_string()).collect();
                write!(f, "{}({})", call.fn_name, arg_strs.join(", "))
            }
            AstKind::CallArg(arg) => match arg.name {
                Some(name) => write!(f, "{} = {}", name, arg.expression.to_string()),
                None => write!(f, "{}", arg.expression.to_string()),
            },
            AstKind::Definition(dfn) => {
                write!(f, "{} = {}", dfn.name, dfn.rhs.to_string(),)
            }
            AstKind::Index(index) => {
                write!(f, "{}[{}]", index.name, index.index)
            }
        }
    }
}
