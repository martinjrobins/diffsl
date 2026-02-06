#[cfg(feature = "external")]
include!("support/external_test_macros.rs");

#[cfg(feature = "external")]
define_external_test!(f64, external_module_compiler_runs_f64);
