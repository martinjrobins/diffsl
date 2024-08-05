use diffsl::{discretise::DiscreteModel, execution::LlvmCompiler, parser::parse_ds_string};
use divan::Bencher;
use ndarray::Array1;

fn main() {
    divan::main();
}

fn setup(n: usize, f_text: &str, name: &str) -> LlvmCompiler {
    let u = vec![1.0; n];
    let full_text = format!(
        "
        u_i {{
            {} 
        }}
        F_i {{
            {}
        }}
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
    LlvmCompiler::from_discrete_model(&discrete_model, out.as_str()).unwrap()
}

#[divan::bench(consts = [1, 10, 100, 1000])]
fn add_scalar_diffsl<const N: usize>(bencher: Bencher) {
    let n = N;
    let compiler = setup(n, "u_i + 1.0", "add_scalar");
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
#[divan::bench(consts = [1, 10, 100, 1000])]
fn add_scalar_ndarray<const N: usize>(bencher: Bencher) {
    let n = N;
    let u = Array1::from_shape_vec((n,), vec![1.0; n]).unwrap();

    bencher.bench_local(|| {
        let _ = &u + 1.0;
    });
}
