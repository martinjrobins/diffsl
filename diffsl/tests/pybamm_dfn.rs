use diffsl::{
    discretise::DiscreteModel, parser::parse_ds_string, CodegenModuleCompile, CodegenModuleJit,
    Compiler,
};
use std::io::Write;

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
    let compiler = Compiler::<M, f64>::from_discrete_model(
        &discrete_model,
        Default::default(),
        Some(full_text.as_str()),
    )
    .unwrap();
    let mut data = compiler.get_new_data();
    let (n_states, n_inputs, _, _, _, _) = compiler.get_dims();
    let inputs = vec![1.0; n_inputs];
    compiler.set_inputs(&inputs, &mut data);
    let mut u = vec![0.0; n_states];
    compiler.set_u0(&mut u, &mut data);
    let mut rr = vec![0.0; n_states];
    let t = 0.0;
    compiler.rhs(t, &u, &mut data, &mut rr);
    let v = vec![1.; n_states];
    let mut drr = vec![0.0; n_states];
    let mut ddata = compiler.get_new_data();
    println!("Computing rhs grad...");
    // flush stdout to ensure the print appears before any potential panic
    std::io::stdout().flush().unwrap();
    compiler.rhs_grad(t, &u, &v, &data, &mut ddata, &rr, &mut drr);
}
