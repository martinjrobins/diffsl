use anyhow::Result;
use target_lexicon::Triple;

use crate::discretise::DiscreteModel;

use super::{compiler::CompilerMode, DataLayout};

pub trait CodegenModule: Sized + Sync {
    type FuncId;
    
    fn from_discrete_model(
        model: &DiscreteModel,
        mode: CompilerMode,
        triple: Option<Triple>,
    ) -> Result<Self> {
        let thread_dim = mode.thread_dim(model.state().nnz());
        let threaded = thread_dim > 1;

        let mut module = Self::new(triple, model, threaded)?;

        let set_u0 = module.compile_set_u0(model)?;
        let _calc_stop = module.compile_calc_stop(model)?;
        let rhs = module.compile_rhs(model)?;
        let mass = module.compile_mass(model)?;
        let calc_out = module.compile_calc_out(model)?;
        let _set_id = module.compile_set_id(model)?;
        let _get_dims = module.compile_get_dims(model)?;
        let set_inputs = module.compile_set_inputs(model)?;
        let _get_inputs = module.compile_get_inputs(model)?;
        let _set_constants = module.compile_set_constants(model)?;
        let tensor_info = module
            .layout()
            .tensors()
            .map(|(name, is_constant)| (name.to_string(), is_constant))
            .collect::<Vec<_>>();
        for (tensor, is_constant) in tensor_info {
            if is_constant {
                module.compile_get_constant(model, tensor.as_str())?;
            } else {
                module.compile_get_tensor(model, tensor.as_str())?;
            }
        }

        module.pre_autodiff_optimisation()?;

        let _set_u0_grad = module.compile_set_u0_grad(&set_u0, model)?;
        let _rhs_grad = module.compile_rhs_grad(&rhs, model)?;
        let _calc_out_grad = module.compile_calc_out_grad(&calc_out, model)?;
        let _set_inputs_grad = module.compile_set_inputs_grad(&set_inputs, model)?;

        if module.supports_reverse_autodiff() {
            module.compile_set_u0_rgrad(&set_u0, model)?;
            module.compile_rhs_rgrad(&rhs, model)?;
            module.compile_calc_out_rgrad(&calc_out, model)?;
            module.compile_set_inputs_rgrad(&set_inputs, model)?;
            module.compile_mass_rgrad(&mass, model)?;

            let rhs_full = module.compile_rhs_full(model)?;
            module.compile_rhs_sgrad(&rhs_full, model)?;
            module.compile_rhs_srgrad(&rhs_full, model)?;
            let calc_out_full = module.compile_calc_out_full(model)?;
            module.compile_calc_out_sgrad(&calc_out_full, model)?;
            module.compile_calc_out_srgrad(&calc_out_full, model)?;
        }

        module.post_autodiff_optimisation()?;
        Ok(module)
    }

    fn new(triple: Option<Triple>, model: &DiscreteModel, threaded: bool) -> Result<Self>;
    fn finish(self) -> Result<Vec<u8>>;

    fn compile_set_u0(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_out(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_out_full(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_calc_stop(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_rhs(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_rhs_full(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_mass(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_get_dims(&mut self, model: &DiscreteModel) -> Result<Self::FuncId>;
    fn compile_get_tensor(&mut self, model: &DiscreteModel, name: &str) -> Result<Self::FuncId>;
    fn compile_get_constant(&mut self, model: &DiscreteModel, name: &str) -> Result<Self::FuncId>;
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

    fn pre_autodiff_optimisation(&mut self) -> Result<()>;
    fn post_autodiff_optimisation(&mut self) -> Result<()>;

    fn layout(&self) -> &DataLayout;
}
