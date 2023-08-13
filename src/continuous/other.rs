struct StateVariable<'a> {
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
    states: Vec<StateVariable>,
    variables: Vec<Variable>,
    ast: Ast<'a>,
}
