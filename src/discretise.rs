
 
#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct Function {
    pub args: Vec<Ast<'a>>,
    pub stmts: Vec<Ast<'a>>,
}
 


#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'a> {
    // F(t, u, u_dot)
    pub f_func: Function,
    // G(t, u)
    pub g_func: Function,
    pub n_states: usize,
}

fn discretise<'s, 'a, 'mi>(model: &'a Ast<'s>, models: I, info: &'mi ModelInfo<'s, 'a>) -> DiscreteModel<'a> {
}
