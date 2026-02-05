use diffsl::execution::compiler::CompilerMode;
use diffsl::{Compiler, ExternalModule};

include!("support/external_test_macros.rs");

define_external_test!(f32, external_module_compiler_runs_f32);
