#[cfg(feature = "external_f32")]
include!("support/external_test_macros.rs");

#[cfg(feature = "external_f32")]
define_external_test!(f32, external_module_compiler_runs_f32);
