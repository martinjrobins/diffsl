use anyhow::Result;
use target_lexicon::Triple;

use crate::discretise::DiscreteModel;

use super::DataLayout;

pub trait CodegenModule: Sized + Sync {
    type FuncId;

    fn new(triple: Triple, model: &DiscreteModel, threaded: bool) -> Result<Self>;
    fn compile_set_u0(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_out(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_out_full(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_stop(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_rhs(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_rhs_full(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_mass(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_get_dims(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_get_tensor(&mut self, model: &DiscreteModel, name: &str) -> Result<Self::FuncId>;
    fn compile_set_inputs(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_get_inputs(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_set_id(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_set_constants(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;

    fn compile_mass_rgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_set_u0_grad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_rhs_grad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_calc_out_grad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;
    fn compile_set_inputs_grad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_set_u0_rgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_rhs_rgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_calc_out_rgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_set_inputs_rgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_rhs_sgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_rhs_srgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_calc_out_sgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn compile_calc_out_srgrad(
        &mut self,
        func_id: &Self::FuncId,
        model: &DiscreteModel,
    ) -> Result<Self::FuncId>;

    fn supports_reverse_autodiff(&self) -> bool;

    fn jit(&mut self, func_id: Self::FuncId) -> Result<*const u8>;
    fn jit_barrier_init(&mut self) -> Result<*const u8>;

    fn get_constants(&self) -> &[f64];

    fn pre_autodiff_optimisation(&mut self) -> Result<()>;
    fn post_autodiff_optimisation(&mut self) -> Result<()>;

    fn layout(&self) -> &DataLayout;
}
