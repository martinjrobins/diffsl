#[cfg(all(feature = "external_dynamic", not(target_arch = "wasm32")))]
include!("support/external_dynamic_test_macros.rs");

#[cfg(all(feature = "external_dynamic", not(target_arch = "wasm32")))]
define_external_dynamic_test!(
    f32,
    external_dynamic_module_compiler_runs_f32,
    "external_dynamic_fixture_f32"
);
