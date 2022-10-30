
#[derive(Debug)]
pub struct DiscreteModel<'a> {
    pub eqns: Vec<Ast<'a>>,
}

fn discretise<'s, 'a, 'mi>(model: &'a Ast<'s>, models: I, info: &'mi ModelInfo<'s, 'a>) -> DiscreteModel<'a> {
}
