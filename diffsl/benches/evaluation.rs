use diffsl::{
    discretise::DiscreteModel,
    execution::module::{CodegenModuleCompile, CodegenModuleJit},
    parser::parse_ds_string,
    Compiler,
};
use ndarray::Array1;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    divan::main();
}

fn setup<M: CodegenModuleCompile + CodegenModuleJit>(
    n: usize,
    f_text: &str,
    name: &str,
) -> Compiler<M, f64> {
    let full_text = format!(
        "
        a_ij {{
            (0..{},1..{}): 1.0,
            (0..{},0..{}): 1.0,
            (1..{},0..{}): 1.0
        }}
        u_i {{
            (0:{}): 1 
        }}
        F_i {{
            {}
        }}
        out_i {{
            u_i 
        }}
        ",
        n - 1,
        n,
        n,
        n,
        n,
        n - 1,
        n,
        f_text,
    );
    let model = parse_ds_string(&full_text).unwrap();
    let discrete_model = DiscreteModel::build(name, &model).unwrap();
    Compiler::<M, f64>::from_discrete_model(&discrete_model, Default::default()).unwrap()
}

fn execute<const N: usize, M: CodegenModuleCompile + CodegenModuleJit>(
    bencher: divan::Bencher,
    f_text: &str,
) {
    let n = N;
    let compiler = setup::<M>(n, f_text, "execute");
    let mut data = compiler.get_new_data();
    compiler.set_inputs(&[], data.as_mut_slice());
    let mut u = vec![1.0; n];
    compiler.set_u0(u.as_mut_slice(), data.as_mut_slice());
    let mut rr = vec![0.0; n];
    let t = 0.0;

    bencher.bench_local(|| {
        compiler.rhs(t, &u, &mut data, &mut rr);
    });
}

#[divan::bench(consts = [2, 10, 100, 1000],sample_count = 1, sample_size = 1)]
fn mat_vec_diffsl_llvm_setup<const N: usize>(bencher: divan::Bencher) {
    bencher.bench_local(|| {
        let n = N;
        setup::<diffsl::LlvmModule>(n, "a_ij * u_j", "setup");
    });
}

#[divan::bench(consts = [2, 10, 100, 1000],sample_count = 1, sample_size = 1)]
fn mat_vec_diffsl_cranelift_setup<const N: usize>(bencher: divan::Bencher) {
    bencher.bench_local(|| {
        let n = N;
        setup::<diffsl::LlvmModule>(n, "a_ij * u_j", "setup");
    });
}

#[cfg(feature = "llvm")]
#[divan::bench(consts = [2, 10, 100, 1000])]
fn mat_vec_diffsl_llvm<const N: usize>(bencher: divan::Bencher) {
    execute::<N, diffsl::LlvmModule>(bencher, "a_ij * u_j");
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cranelift"))]
#[divan::bench(consts = [2, 10, 100, 1000])]
fn mat_vec_diffsl_cranelift<const N: usize>(bencher: divan::Bencher) {
    execute::<N, diffsl::CraneliftJitModule>(bencher, "a_ij * u_j");
}

#[cfg(feature = "llvm")]
#[divan::bench(consts = [2, 10, 100, 1000, 2000, 4000])]
fn add_scalar_diffsl_llvm<const N: usize>(bencher: divan::Bencher) {
    execute::<N, diffsl::LlvmModule>(bencher, "u_i + 1.0");
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cranelift"))]
#[divan::bench(consts = [2, 10, 100, 1000, 2000, 4000])]
fn add_scalar_diffsl_cranelift<const N: usize>(bencher: divan::Bencher) {
    execute::<N, diffsl::CraneliftJitModule>(bencher, "u_i + 1.0");
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench(consts = [2, 10, 100, 1000, 2000, 4000])]
fn add_scalar_ndarray<const N: usize>(bencher: divan::Bencher) {
    let n = N;
    let u = Array1::from_shape_vec((n,), vec![1.0; n]).unwrap();

    bencher.bench_local(|| {
        let _ = &u + 1.0;
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench(consts = [2, 10, 100, 1000, 2000, 4000])]
fn mat_vec_faer<const N: usize>(bencher: divan::Bencher) {
    let n = N;
    let u = faer::Col::<f64>::from_fn(n, |_i| 1.0);
    let triplets = (0..n)
        .flat_map(|i| {
            let mut row = vec![faer::sparse::Triplet::new(i, i, 0.0)];
            if i + 1 < n {
                row.push(faer::sparse::Triplet::new(i, i + 1, 1.0));
            }
            if i >= 1 {
                row.push(faer::sparse::Triplet::new(i, i - 1, 1.0));
            }
            row
        })
        .collect::<Vec<_>>();
    let a = faer::sparse::SparseRowMat::try_new_from_triplets(n, n, triplets.as_slice()).unwrap();

    bencher.bench_local(|| {
        let _ = &a * &u;
    });
}
