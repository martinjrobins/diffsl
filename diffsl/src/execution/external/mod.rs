use std::{collections::HashMap, marker::PhantomData};

use anyhow::Result;

use super::module::{CodegenModule, CodegenModuleJit};

type UIntType = u32;

mod f32_symbols;
mod f64_symbols;

pub struct ExternalModule<T> {
    _marker: PhantomData<T>,
}

impl<T> ExternalModule<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for ExternalModule<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CodegenModule for ExternalModule<T> where T: Send + Sync + 'static {}

trait ExternSymbols {
    fn insert_symbols(symbols: &mut HashMap<String, *const u8>);
}

impl<T> CodegenModuleJit for ExternalModule<T>
where
    T: ExternSymbols + Send + Sync + 'static,
{
    fn jit(&mut self) -> Result<HashMap<String, *const u8>> {
        let mut symbols = HashMap::new();
        T::insert_symbols(&mut symbols);
        Ok(symbols)
    }
}

macro_rules! impl_extern_symbols {
    ($ty:ty, $sym:path, { $($name:literal => $func:ident),+ $(,)? }) => {
        impl ExternSymbols for $ty {
            fn insert_symbols(symbols: &mut HashMap<String, *const u8>) {
                use $sym as sym;
                $(symbols.insert($name.to_string(), sym::$func as *const u8);)+
            }
        }
    };
}

impl_extern_symbols!(f64, f64_symbols, {
    "barrier_init" => barrier_init_f64,
    "set_constants" => set_constants_f64,
    "set_u0" => set_u0_f64,
    "rhs" => rhs_f64,
    "rhs_grad" => rhs_grad_f64,
    "rhs_rgrad" => rhs_rgrad_f64,
    "rhs_sgrad" => rhs_sgrad_f64,
    "rhs_srgrad" => rhs_srgrad_f64,
    "mass" => mass_f64,
    "mass_rgrad" => mass_rgrad_f64,
    "set_u0_grad" => set_u0_grad_f64,
    "set_u0_rgrad" => set_u0_rgrad_f64,
    "set_u0_sgrad" => set_u0_sgrad_f64,
    "calc_out" => calc_out_f64,
    "calc_out_grad" => calc_out_grad_f64,
    "calc_out_rgrad" => calc_out_rgrad_f64,
    "calc_out_sgrad" => calc_out_sgrad_f64,
    "calc_out_srgrad" => calc_out_srgrad_f64,
    "calc_stop" => calc_stop_f64,
    "set_id" => set_id_f64,
    "get_dims" => get_dims_f64,
    "set_inputs" => set_inputs_f64,
    "get_inputs" => get_inputs_f64,
    "set_inputs_grad" => set_inputs_grad_f64,
    "set_inputs_rgrad" => set_inputs_rgrad_f64,
});

impl_extern_symbols!(f32, f32_symbols, {
    "barrier_init" => barrier_init_f32,
    "set_constants" => set_constants_f32,
    "set_u0" => set_u0_f32,
    "rhs" => rhs_f32,
    "rhs_grad" => rhs_grad_f32,
    "rhs_rgrad" => rhs_rgrad_f32,
    "rhs_sgrad" => rhs_sgrad_f32,
    "rhs_srgrad" => rhs_srgrad_f32,
    "mass" => mass_f32,
    "mass_rgrad" => mass_rgrad_f32,
    "set_u0_grad" => set_u0_grad_f32,
    "set_u0_rgrad" => set_u0_rgrad_f32,
    "set_u0_sgrad" => set_u0_sgrad_f32,
    "calc_out" => calc_out_f32,
    "calc_out_grad" => calc_out_grad_f32,
    "calc_out_rgrad" => calc_out_rgrad_f32,
    "calc_out_sgrad" => calc_out_sgrad_f32,
    "calc_out_srgrad" => calc_out_srgrad_f32,
    "calc_stop" => calc_stop_f32,
    "set_id" => set_id_f32,
    "get_dims" => get_dims_f32,
    "set_inputs" => set_inputs_f32,
    "get_inputs" => get_inputs_f32,
    "set_inputs_grad" => set_inputs_grad_f32,
    "set_inputs_rgrad" => set_inputs_rgrad_f32,
});

#[cfg(test)]
mod tests {
    use crate::execution::compiler::CompilerMode;
    use crate::{Compiler, ExternalModule};

    include!("../../../tests/support/external_test_macros.rs");

    define_external_test!(f64, external_module_compiler_runs);
}
