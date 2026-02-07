use std::{collections::HashMap, marker::PhantomData};

use anyhow::Result;

use super::module::{CodegenModule, CodegenModuleJit};

type UIntType = u32;

macro_rules! define_symbol_module {
    ($mod_name:ident, $ty:ty) => {
        mod $mod_name {
            use super::UIntType;

            #[allow(clashing_extern_declarations)]
            extern "C" {
                #[link_name = "barrier_init"]
                pub fn barrier_init();
                #[link_name = "set_constants"]
                pub fn set_constants(thread_id: UIntType, thread_dim: UIntType);
                #[link_name = "set_u0"]
                pub fn set_u0(
                    u: *mut $ty,
                    data: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "rhs"]
                pub fn rhs(
                    time: $ty,
                    u: *const $ty,
                    data: *mut $ty,
                    rr: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "rhs_grad"]
                pub fn rhs_grad(
                    time: $ty,
                    u: *const $ty,
                    du: *const $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    rr: *const $ty,
                    drr: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "rhs_rgrad"]
                pub fn rhs_rgrad(
                    time: $ty,
                    u: *const $ty,
                    du: *mut $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    rr: *const $ty,
                    drr: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "rhs_sgrad"]
                pub fn rhs_sgrad(
                    time: $ty,
                    u: *const $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    rr: *const $ty,
                    drr: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "rhs_srgrad"]
                pub fn rhs_srgrad(
                    time: $ty,
                    u: *const $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    rr: *const $ty,
                    drr: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "mass"]
                pub fn mass(
                    time: $ty,
                    v: *const $ty,
                    data: *mut $ty,
                    mv: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "mass_rgrad"]
                pub fn mass_rgrad(
                    time: $ty,
                    v: *const $ty,
                    dv: *mut $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    mv: *const $ty,
                    dmv: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "set_u0_grad"]
                pub fn set_u0_grad(
                    u: *const $ty,
                    du: *mut $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "set_u0_rgrad"]
                pub fn set_u0_rgrad(
                    u: *const $ty,
                    du: *mut $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "set_u0_sgrad"]
                pub fn set_u0_sgrad(
                    u: *const $ty,
                    du: *mut $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "calc_out"]
                pub fn calc_out(
                    time: $ty,
                    u: *const $ty,
                    data: *mut $ty,
                    out: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "calc_out_grad"]
                pub fn calc_out_grad(
                    time: $ty,
                    u: *const $ty,
                    du: *const $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    out: *const $ty,
                    dout: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "calc_out_rgrad"]
                pub fn calc_out_rgrad(
                    time: $ty,
                    u: *const $ty,
                    du: *mut $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    out: *const $ty,
                    dout: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "calc_out_sgrad"]
                pub fn calc_out_sgrad(
                    time: $ty,
                    u: *const $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    out: *const $ty,
                    dout: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "calc_out_srgrad"]
                pub fn calc_out_srgrad(
                    time: $ty,
                    u: *const $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                    out: *const $ty,
                    dout: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "calc_stop"]
                pub fn calc_stop(
                    time: $ty,
                    u: *const $ty,
                    data: *mut $ty,
                    root: *mut $ty,
                    thread_id: UIntType,
                    thread_dim: UIntType,
                );
                #[link_name = "set_id"]
                pub fn set_id(id: *mut $ty);
                #[link_name = "get_dims"]
                pub fn get_dims(
                    states: *mut UIntType,
                    inputs: *mut UIntType,
                    outputs: *mut UIntType,
                    data: *mut UIntType,
                    stop: *mut UIntType,
                    has_mass: *mut UIntType,
                );
                #[link_name = "set_inputs"]
                pub fn set_inputs(inputs: *const $ty, data: *mut $ty);
                #[link_name = "get_inputs"]
                pub fn get_inputs(inputs: *mut $ty, data: *const $ty);
                #[link_name = "set_inputs_grad"]
                pub fn set_inputs_grad(
                    inputs: *const $ty,
                    dinputs: *const $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                );
                #[link_name = "set_inputs_rgrad"]
                pub fn set_inputs_rgrad(
                    inputs: *const $ty,
                    dinputs: *mut $ty,
                    data: *const $ty,
                    ddata: *mut $ty,
                );
            }
        }
    };
}

define_symbol_module!(f32_symbols, f32);
define_symbol_module!(f64_symbols, f64);

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

pub trait ExternSymbols {
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
    "barrier_init" => barrier_init,
    "set_constants" => set_constants,
    "set_u0" => set_u0,
    "rhs" => rhs,
    "rhs_grad" => rhs_grad,
    "rhs_rgrad" => rhs_rgrad,
    "rhs_sgrad" => rhs_sgrad,
    "rhs_srgrad" => rhs_srgrad,
    "mass" => mass,
    "mass_rgrad" => mass_rgrad,
    "set_u0_grad" => set_u0_grad,
    "set_u0_rgrad" => set_u0_rgrad,
    "set_u0_sgrad" => set_u0_sgrad,
    "calc_out" => calc_out,
    "calc_out_grad" => calc_out_grad,
    "calc_out_rgrad" => calc_out_rgrad,
    "calc_out_sgrad" => calc_out_sgrad,
    "calc_out_srgrad" => calc_out_srgrad,
    "calc_stop" => calc_stop,
    "set_id" => set_id,
    "get_dims" => get_dims,
    "set_inputs" => set_inputs,
    "get_inputs" => get_inputs,
    "set_inputs_grad" => set_inputs_grad,
    "set_inputs_rgrad" => set_inputs_rgrad,
});

impl_extern_symbols!(f32, f32_symbols, {
    "barrier_init" => barrier_init,
    "set_constants" => set_constants,
    "set_u0" => set_u0,
    "rhs" => rhs,
    "rhs_grad" => rhs_grad,
    "rhs_rgrad" => rhs_rgrad,
    "rhs_sgrad" => rhs_sgrad,
    "rhs_srgrad" => rhs_srgrad,
    "mass" => mass,
    "mass_rgrad" => mass_rgrad,
    "set_u0_grad" => set_u0_grad,
    "set_u0_rgrad" => set_u0_rgrad,
    "set_u0_sgrad" => set_u0_sgrad,
    "calc_out" => calc_out,
    "calc_out_grad" => calc_out_grad,
    "calc_out_rgrad" => calc_out_rgrad,
    "calc_out_sgrad" => calc_out_sgrad,
    "calc_out_srgrad" => calc_out_srgrad,
    "calc_stop" => calc_stop,
    "set_id" => set_id,
    "get_dims" => get_dims,
    "set_inputs" => set_inputs,
    "get_inputs" => get_inputs,
    "set_inputs_grad" => set_inputs_grad,
    "set_inputs_rgrad" => set_inputs_rgrad,
});
