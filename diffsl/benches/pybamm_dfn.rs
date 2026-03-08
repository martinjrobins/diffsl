use diffsl::{
    discretise::DiscreteModel, parser::parse_ds_string, CodegenModuleCompile, CodegenModuleJit,
    Compiler,
};
use divan::Bencher;

fn main() {
    divan::main();
}

fn pybamm_dfn_execute_rhs_grad<M: CodegenModuleCompile + CodegenModuleJit>(bencher: Bencher) {
    let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
    let model = parse_ds_string(&full_text).unwrap();
    let discrete_model = DiscreteModel::build("pybamm_dfn", &model).unwrap();
    let compiler = Compiler::<M, f64>::from_discrete_model(
        &discrete_model,
        Default::default(),
        Some(full_text.as_str()),
    )
    .unwrap();
    let (n_states, _n_inputs, _n_outputs, _n_data, _n_stop, _has_mass) = compiler.get_dims();
    let t = 0.0;
    let y = vec![1.0; n_states];
    let dy = vec![0.1; n_states];
    let mut data = compiler.get_new_data();
    let mut ddata = compiler.get_new_data();
    let mut rr = vec![0.0; n_states];
    let mut drr = vec![0.0; n_states];
    bencher.bench_local(move || {
        compiler.rhs_grad(
            t,
            y.as_slice(),
            dy.as_slice(),
            data.as_mut_slice(),
            ddata.as_mut_slice(),
            rr.as_mut_slice(),
            drr.as_mut_slice(),
        );
    });
}

#[cfg(feature = "llvm")]
#[divan::bench]
fn pybamm_dfn_execute_rhs_grad_llvm(bencher: Bencher) {
    pybamm_dfn_execute_rhs_grad::<diffsl::LlvmModule>(bencher);
}

#[cfg(feature = "cranelift")]
#[divan::bench]
fn pybamm_dfn_execute_rhs_grad_cranelift(bencher: Bencher) {
    pybamm_dfn_execute_rhs_grad::<diffsl::CraneliftJitModule>(bencher);
}

fn pybamm_dfn_execute_rhs<M: CodegenModuleCompile + CodegenModuleJit>(bencher: Bencher) {
    let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
    let model = parse_ds_string(&full_text).unwrap();
    let discrete_model = DiscreteModel::build("pybamm_dfn", &model).unwrap();
    let compiler = Compiler::<M, f64>::from_discrete_model(
        &discrete_model,
        Default::default(),
        Some(full_text.as_str()),
    )
    .unwrap();
    let (n_states, _n_inputs, _n_outputs, _n_data, _n_stop, _has_mass) = compiler.get_dims();
    let t = 0.0;
    let y = vec![1.0; n_states];
    let mut data = compiler.get_new_data();
    let mut rr = vec![0.0; n_states];
    bencher.bench_local(move || {
        compiler.rhs(t, y.as_slice(), data.as_mut_slice(), rr.as_mut_slice());
    });
}

#[cfg(feature = "llvm")]
#[divan::bench]
fn pybamm_dfn_execute_rhs_llvm(bencher: Bencher) {
    pybamm_dfn_execute_rhs::<diffsl::LlvmModule>(bencher);
}

#[cfg(feature = "cranelift")]
#[divan::bench]
fn pybamm_dfn_execute_rhs_cranelift(bencher: Bencher) {
    pybamm_dfn_execute_rhs::<diffsl::CraneliftJitModule>(bencher);
}

fn pybamm_dfn_compile<M: CodegenModuleCompile + CodegenModuleJit>(bencher: Bencher) {
    let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
    let model = parse_ds_string(&full_text).unwrap();
    let discrete_model = DiscreteModel::build("pybamm_dfn", &model).unwrap();
    bencher.bench_local(move || {
        Compiler::<M, f64>::from_discrete_model(&discrete_model, Default::default(), None).unwrap();
    });
}

#[cfg(feature = "llvm")]
#[divan::bench(sample_count = 1, sample_size = 1)]
fn pybamm_dfn_compile_llvm(bencher: Bencher) {
    pybamm_dfn_compile::<diffsl::LlvmModule>(bencher);
}

#[cfg(feature = "cranelift")]
#[divan::bench(sample_count = 10, sample_size = 1)]
fn pybamm_dfn_compile_cranelift(bencher: Bencher) {
    pybamm_dfn_compile::<diffsl::CraneliftJitModule>(bencher);
}

#[divan::bench]
fn pybamm_dfn_parse(bencher: Bencher) {
    let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
    bencher.bench_local(move || {
        parse_ds_string(&full_text).unwrap();
    });
}

#[divan::bench]
fn pybamm_dfn_build(bencher: Bencher) {
    let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
    let model = parse_ds_string(&full_text).unwrap();
    bencher.bench_local(move || {
        DiscreteModel::build("pybamm_dfn", &model).unwrap();
    });
}
