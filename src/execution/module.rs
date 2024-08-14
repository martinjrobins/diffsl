use target_lexicon::Triple;
use anyhow::Result;

use crate::discretise::DiscreteModel;

use super::DataLayout;


pub trait CodeGenModule {
    type FuncId;

    fn new(triple: Triple, model: &DiscreteModel) -> Self;
    fn compile_set_u0(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_out(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_stop(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_rhs(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_mass(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_get_dims(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_get_tensor(&mut self, model: &DiscreteModel, name: &str) -> Result<Self::FuncId>;
    fn compile_set_inputs(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_set_id(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;

    fn compile_set_u0_grad(&mut self, func_id: Self::FuncId) -> Result<Self::FuncId>;
    fn compile_rhs_grad(&mut self, func_id: Self::FuncId) -> Result<Self::FuncId>;
    fn compile_calc_out_grad(&mut self, func_id: Self::FuncId) -> Result<Self::FuncId>;
    fn compile_set_inputs_grad(&mut self, func_id: Self::FuncId) -> Result<Self::FuncId>;

    fn jit<O>(&mut self, func_id: Self::FuncId) -> Result<O>;

    fn pre_autodiff_optimisation(&mut self) -> Result<()>;
    fn post_autodiff_optimisation(&mut self) -> Result<()>;

    fn layout(&self) -> &DataLayout;
}


