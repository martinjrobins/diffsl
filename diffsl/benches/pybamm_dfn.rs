use diffsl::{CodegenModuleCompile, CodegenModuleJit, Compiler, discretise::DiscreteModel, parser::parse_ds_string};
use divan::Bencher;

fn main() {
    divan::main();
}

fn pybamm_dfn_compile<M: CodegenModuleCompile + CodegenModuleJit>(bencher: Bencher) {
    let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
    let model = parse_ds_string(&full_text).unwrap();
    let discrete_model = DiscreteModel::build("pybamm_dfn", &model).unwrap();
    bencher.bench_local(move || {
        Compiler::<M, f64>::from_discrete_model(&discrete_model, Default::default()).unwrap();
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