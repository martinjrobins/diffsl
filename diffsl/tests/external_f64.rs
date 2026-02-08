#[cfg(feature = "external_f64")]
include!("support/external_test_macros.rs");

#[cfg(feature = "external_f64")]
define_external_test!(f64, external_module_compiler_runs_f64);
