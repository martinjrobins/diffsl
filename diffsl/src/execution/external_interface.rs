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

#[allow(unused_macros)]
macro_rules! insert_external_symbols {
    ($symbols:expr, $sym:path) => {{
        use $sym as sym;
        $symbols.insert("barrier_init".to_string(), sym::barrier_init as *const u8);
        $symbols.insert("set_constants".to_string(), sym::set_constants as *const u8);
        $symbols.insert("set_u0".to_string(), sym::set_u0 as *const u8);
        $symbols.insert("reset".to_string(), sym::reset as *const u8);
        $symbols.insert("reset_grad".to_string(), sym::reset_grad as *const u8);
        $symbols.insert("reset_rgrad".to_string(), sym::reset_rgrad as *const u8);
        $symbols.insert("reset_sgrad".to_string(), sym::reset_sgrad as *const u8);
        $symbols.insert("reset_srgrad".to_string(), sym::reset_srgrad as *const u8);
        $symbols.insert("rhs".to_string(), sym::rhs as *const u8);
        $symbols.insert("rhs_grad".to_string(), sym::rhs_grad as *const u8);
        $symbols.insert("rhs_rgrad".to_string(), sym::rhs_rgrad as *const u8);
        $symbols.insert("rhs_sgrad".to_string(), sym::rhs_sgrad as *const u8);
        $symbols.insert("rhs_srgrad".to_string(), sym::rhs_srgrad as *const u8);
        $symbols.insert("mass".to_string(), sym::mass as *const u8);
        $symbols.insert("mass_rgrad".to_string(), sym::mass_rgrad as *const u8);
        $symbols.insert("set_u0_grad".to_string(), sym::set_u0_grad as *const u8);
        $symbols.insert("set_u0_rgrad".to_string(), sym::set_u0_rgrad as *const u8);
        $symbols.insert("set_u0_sgrad".to_string(), sym::set_u0_sgrad as *const u8);
        $symbols.insert("calc_out".to_string(), sym::calc_out as *const u8);
        $symbols.insert("calc_out_grad".to_string(), sym::calc_out_grad as *const u8);
        $symbols.insert(
            "calc_out_rgrad".to_string(),
            sym::calc_out_rgrad as *const u8,
        );
        $symbols.insert(
            "calc_out_sgrad".to_string(),
            sym::calc_out_sgrad as *const u8,
        );
        $symbols.insert(
            "calc_out_srgrad".to_string(),
            sym::calc_out_srgrad as *const u8,
        );
        $symbols.insert("calc_stop".to_string(), sym::calc_stop as *const u8);
        $symbols.insert(
            "calc_stop_grad".to_string(),
            sym::calc_stop_grad as *const u8,
        );
        $symbols.insert(
            "calc_stop_rgrad".to_string(),
            sym::calc_stop_rgrad as *const u8,
        );
        $symbols.insert(
            "calc_stop_sgrad".to_string(),
            sym::calc_stop_sgrad as *const u8,
        );
        $symbols.insert(
            "calc_stop_srgrad".to_string(),
            sym::calc_stop_srgrad as *const u8,
        );
        $symbols.insert("set_id".to_string(), sym::set_id as *const u8);
        $symbols.insert("get_dims".to_string(), sym::get_dims as *const u8);
        $symbols.insert("set_inputs".to_string(), sym::set_inputs as *const u8);
        $symbols.insert("get_inputs".to_string(), sym::get_inputs as *const u8);
        $symbols.insert(
            "set_inputs_grad".to_string(),
            sym::set_inputs_grad as *const u8,
        );
        $symbols.insert(
            "set_inputs_rgrad".to_string(),
            sym::set_inputs_rgrad as *const u8,
        );
    }};
}

#[allow(unused_imports)]
pub(crate) use insert_external_symbols;

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
