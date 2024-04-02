use diffsl::{discretise::DiscreteModel, execution::Compiler, parser::parse_ds_string};
use divan::Bencher;

fn main() {
    divan::main();
}

fn setup(n: usize, f_text: &str, name: &str) -> Compiler {
    let u = vec![1.0; n];
    let full_text = format!(
        "
        u_i {{
            {} 
        }}
        {} 
        out_i {{
            u_i 
        }}
        ",
        (0..n)
            .map(|i| format!("x{} = {},", i, u[i]))
            .collect::<Vec<_>>()
            .join("\n"),
        f_text,
    );
    let model = parse_ds_string(&full_text).unwrap();
    let discrete_model = DiscreteModel::build(name, &model).unwrap();
    let out = format!("test_output/benches_evaluation_{}", name);
    Compiler::from_discrete_model(&discrete_model, out.as_str()).unwrap()
}

/// This macro generates benchmarks for the given function.
/// The function should be a diffsl string that includes
/// the rhs F_i { ... }.
macro_rules! bench {
    ($name:ident, $f:expr, $c:expr) => {
        #[divan::bench(consts = $c)]
        fn $name<const N: usize>(bencher: Bencher) {
            let n = N;
            let text = $f.to_string().replace("n", &n.to_string());
            let compiler = setup(n, text.as_str(), stringify!($name));
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
    };
    () => {};
}

bench!(add_scalar, "F_i { u_i + 1.0 }", [1, 10, 100, 1000]);

#[divan::bench(consts = [1, 10, 100, 1000])]
fn add_scalar_rust<const N: usize>(bencher: Bencher) {
    let n = N;
    let u = vec![1.0; n];
    let mut r = vec![0.0; n];
    bencher.bench_local(|| {
        r.iter_mut().zip(u.iter()).for_each(|(r, u)| {
            *r = u + 1.0;
        });
    });
}

bench!(inner_product, "F_i { (0) }", [1, 10, 100, 1000]);

bench!(
    dense_matmul,
    "A_ij { (0:n, 0:n): 1 } F_i { A_ij * u_i }",
    [10, 100, 1000]
);

#[divan::bench(consts = [10, 100, 1000])]
fn dense_matmul_rust<const N: usize>(bencher: Bencher) {
    let n = N;
    let u = vec![1.0; n];
    let a = vec![1.0; n * n];
    let mut r = vec![0.0; n];
    bencher.bench_local(|| {
        r.iter_mut().enumerate().for_each(|(i, r)| {
            *r = (0..n).map(|j| a[i * n + j] * u[j]).sum();
        });
    });
}
