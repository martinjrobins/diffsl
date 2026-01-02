use diffsl::{
    discretise::DiscreteModel, parser::parse_ds_string, CodegenModuleCompile, CodegenModuleJit,
    Compiler,
};

#[cfg(all(feature = "llvm", not(feature = "llvm21-1"), not(feature = "llvm20-1")))]
#[test]
fn test_dfn_model_initialization_llvm() {
    test_dfn_model_initialization::<diffsl::LlvmModule>();
}

#[cfg(feature = "cranelift")]
#[test]
fn test_dfn_model_initialization_cranelift() {
    test_dfn_model_initialization::<diffsl::CraneliftJitModule>();
}

#[allow(dead_code)]
fn test_dfn_model_initialization<M: CodegenModuleJit + CodegenModuleCompile>() {
    let _ = env_logger::builder().is_test(true).try_init();
    let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
    let model = parse_ds_string(&full_text).unwrap();
    let discrete_model = DiscreteModel::build("pybamm_dfn", &model).unwrap();
    Compiler::<M, f64>::from_discrete_model(&discrete_model, Default::default()).unwrap();
}
