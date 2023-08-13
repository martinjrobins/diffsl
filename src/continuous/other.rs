use crate::ast::Ast;

pub struct StateVariable<'a> {
    name: String,
    def: Ast<'a>,
    expr: Ast<'a>,
}

struct Variable<'a> {
    name: String,
    def: Ast<'a>,
    expr: Ast<'a>,
}

struct Model<'a> {
    name: String,
    states: Vec<StateVariable<'a>>,
    variables: Vec<Variable<'a>>,
    ast: Ast<'a>,
}
