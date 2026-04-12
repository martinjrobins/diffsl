pub(crate) const TENSOR_SYMBOL_PREFIX: &str = "get_tensor_";
pub(crate) const CONSTANT_SYMBOL_PREFIX: &str = "get_constant_";

macro_rules! for_each_external_symbol {
    ($callback:ident) => {
        $callback! {
            "barrier_init" => barrier_init,
            "set_constants" => set_constants,
            "set_u0" => set_u0,
            "reset" => reset,
            "reset_grad" => reset_grad,
            "reset_rgrad" => reset_rgrad,
            "reset_sgrad" => reset_sgrad,
            "reset_srgrad" => reset_srgrad,
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
            "calc_stop_grad" => calc_stop_grad,
            "calc_stop_rgrad" => calc_stop_rgrad,
            "calc_stop_sgrad" => calc_stop_sgrad,
            "calc_stop_srgrad" => calc_stop_srgrad,
            "set_id" => set_id,
            "get_dims" => get_dims,
            "set_inputs" => set_inputs,
            "get_inputs" => get_inputs,
            "set_inputs_grad" => set_inputs_grad,
            "set_inputs_rgrad" => set_inputs_rgrad,
        }
    };
}

#[allow(unused_imports)]
pub(crate) use for_each_external_symbol;

macro_rules! collect_external_symbol_names {
    ($($name:literal => $func:ident,)+) => {
        pub(crate) const EXTERNAL_SYMBOL_NAMES: &[&str] = &[$($name),+];
    };
}

for_each_external_symbol!(collect_external_symbol_names);

pub(crate) fn normalize_symbol_name(name: &str) -> &str {
    name.strip_prefix('_').unwrap_or(name)
}

pub(crate) fn is_external_symbol_name(name: &str) -> bool {
    let name = normalize_symbol_name(name);
    EXTERNAL_SYMBOL_NAMES.contains(&name)
        || name.starts_with(TENSOR_SYMBOL_PREFIX)
        || name.starts_with(CONSTANT_SYMBOL_PREFIX)
}
