use std::{
    path::PathBuf,
    process::Command,
};

#[allow(unused_imports)]
use diffsl::execution::compiler::CompilerMode;
#[allow(unused_imports)]
use diffsl::{Compiler, ExternalDynModule};

pub fn build_fixture_library(fixture_name: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest_path = manifest_dir
        .join("tests")
        .join("fixtures")
        .join(fixture_name)
        .join("Cargo.toml");
    let target_dir = manifest_dir
        .join("target")
        .join("external_dynamic_fixtures")
        .join(fixture_name);

    let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
        .arg("build")
        .arg("--manifest-path")
        .arg(&manifest_path)
        .arg("--quiet")
        .env("CARGO_TARGET_DIR", &target_dir)
        .status()
        .expect("fixture crate should build");
    assert!(status.success(), "fixture crate build failed");

    let library_path = target_dir.join("debug").join(dynamic_library_name(fixture_name));
    assert!(
        library_path.exists(),
        "fixture library was not produced at {}",
        library_path.display()
    );
    library_path
}

fn dynamic_library_name(crate_name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{crate_name}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{crate_name}.dylib")
    } else {
        format!("lib{crate_name}.so")
    }
}

#[allow(unused_macros)]
macro_rules! define_external_dynamic_test {
    ($ty:ty, $test_name:ident, $fixture_name:literal) => {
        #[test]
        fn $test_name() {
            const STATES: usize = 1;
            const INPUTS: usize = 1;
            const OUTPUTS: usize = 1;
            const DATA: usize = 1;
            const STOP: usize = 1;

            let fixture_path = build_fixture_library($fixture_name);
            let module = ExternalDynModule::<$ty>::new(&fixture_path)
                .expect("dynamic module should load fixture library");
            let compiler = Compiler::from_codegen_module(module, CompilerMode::SingleThreaded)
                .expect("compiler should build");

            let (n_states, n_inputs, n_outputs, n_data, n_stop, has_mass, has_reset) =
                compiler.get_dims();
            assert_eq!(n_states, STATES);
            assert_eq!(n_inputs, INPUTS);
            assert_eq!(n_outputs, OUTPUTS);
            assert_eq!(n_data, DATA);
            assert_eq!(n_stop, STOP);
            assert!(has_mass);
            assert!(has_reset);

            let mut data = vec![-1.0 as $ty; n_data];
            let inputs = vec![1.0 as $ty; n_inputs];
            compiler.set_inputs(&inputs, &mut data, 0);

            let mut tensors = compiler.get_tensors();
            tensors.sort();
            assert_eq!(tensors, vec!["in".to_string()]);
            assert_eq!(
                compiler.get_tensor_data("in", &data).expect("tensor export should resolve"),
                inputs.as_slice()
            );

            let mut constants = compiler.get_constants();
            constants.sort();
            assert_eq!(constants, vec!["rate".to_string()]);
            assert_eq!(
                compiler
                    .get_constants_data("rate")
                    .expect("constant export should resolve"),
                [3.0 as $ty].as_slice()
            );

            let mut inputs_out = vec![-2.0 as $ty; n_inputs];
            compiler.get_inputs(&mut inputs_out, &data);
            assert_eq!(inputs_out, inputs);

            let mut u = vec![-2.0 as $ty; n_states];
            compiler.set_u0(&mut u, &mut data);
            assert_eq!(u[0], 1.0 as $ty);

            let mut out = vec![-3.0 as $ty; n_outputs];
            compiler.calc_out(0.0 as $ty, &u, &mut data, &mut out);
            assert_eq!(out[0], u[0]);

            let mut rr = vec![-4.0 as $ty; n_states];
            compiler.rhs(0.0 as $ty, &u, &mut data, &mut rr);
            assert_eq!(rr[0], 0.0 as $ty);

            let mut stop = vec![-5.0 as $ty; n_stop];
            compiler.calc_stop(0.0 as $ty, &u, &mut data, &mut stop);
            assert_eq!(stop[0], 0.5 as $ty);

            let du_stop = vec![1.0 as $ty; n_states];
            let mut ddata_stop = vec![-5.15 as $ty; n_data];
            let mut dstop = vec![-5.25 as $ty; n_stop];
            compiler.calc_stop_grad(
                0.0 as $ty,
                &u,
                &du_stop,
                &data,
                &mut ddata_stop,
                &stop,
                &mut dstop,
            );
            assert_eq!(dstop[0], 1.0 as $ty);

            let mut du_stop_rev = vec![-5.35 as $ty; n_states];
            let mut ddata_stop_rev = vec![-5.45 as $ty; n_data];
            let mut dstop_rev = vec![1.0 as $ty; n_stop];
            compiler.calc_stop_rgrad(
                0.0 as $ty,
                &u,
                &mut du_stop_rev,
                &data,
                &mut ddata_stop_rev,
                &stop,
                &mut dstop_rev,
            );
            assert!((du_stop_rev[0] - (-4.35 as $ty)).abs() < (1e-6 as $ty));

            let mut ddata_stop_s = vec![-5.55 as $ty; n_data];
            let mut dstop_s = vec![-5.65 as $ty; n_stop];
            compiler.calc_stop_sgrad(
                0.0 as $ty,
                &u,
                &data,
                &mut ddata_stop_s,
                &stop,
                &mut dstop_s,
            );
            assert_eq!(dstop_s[0], 0.0 as $ty);

            let mut ddata_stop_sr = vec![-5.75 as $ty; n_data];
            let mut dstop_sr = vec![1.0 as $ty; n_stop];
            compiler.calc_stop_srgrad(
                0.0 as $ty,
                &u,
                &data,
                &mut ddata_stop_sr,
                &stop,
                &mut dstop_sr,
            );
            assert_eq!(dstop_sr[0], 0.0 as $ty);

            let mut reset = vec![-5.5 as $ty; n_states];
            compiler.reset(0.0 as $ty, &u, &mut data, &mut reset);
            assert_eq!(reset[0], 2.0 as $ty);

            let du = vec![1.0 as $ty; n_states];
            let mut ddata = vec![-8.0 as $ty; n_data];
            let mut dreset = vec![-5.75 as $ty; n_states];
            compiler.reset_grad(0.0 as $ty, &u, &du, &data, &mut ddata, &reset, &mut dreset);
            assert_eq!(dreset[0], 2.0 as $ty);

            let mut du_reset_rev = vec![-5.85 as $ty; n_states];
            let mut ddata_reset_rev = vec![-5.95 as $ty; n_data];
            let mut dreset_rev = vec![1.0 as $ty; n_states];
            compiler.reset_rgrad(
                0.0 as $ty,
                &u,
                &mut du_reset_rev,
                &data,
                &mut ddata_reset_rev,
                &reset,
                &mut dreset_rev,
            );
            assert!((du_reset_rev[0] - (-3.85 as $ty)).abs() < (1e-6 as $ty));

            let mut ddata_reset_s = vec![-6.05 as $ty; n_data];
            let mut dreset_s = vec![-6.15 as $ty; n_states];
            compiler.reset_sgrad(
                0.0 as $ty,
                &u,
                &data,
                &mut ddata_reset_s,
                &reset,
                &mut dreset_s,
            );
            assert_eq!(dreset_s[0], 0.0 as $ty);

            let mut ddata_reset_sr = vec![-6.25 as $ty; n_data];
            let mut dreset_sr = vec![1.0 as $ty; n_states];
            compiler.reset_srgrad(
                0.0 as $ty,
                &u,
                &data,
                &mut ddata_reset_sr,
                &reset,
                &mut dreset_sr,
            );
            assert_eq!(dreset_sr[0], 0.0 as $ty);

            let mut mv = vec![-6.0 as $ty; n_states];
            compiler.mass(0.0 as $ty, &u, &mut data, &mut mv);
            assert_eq!(mv[0], 1.0 as $ty);

            let mut id = vec![-7.0 as $ty; n_states];
            compiler.set_id(&mut id);
            assert_eq!(id[0], 42.0 as $ty);

            let mut drr = vec![-9.0 as $ty; n_states];
            compiler.rhs_grad(0.0 as $ty, &u, &du, &data, &mut ddata, &rr, &mut drr);
            assert_eq!(drr[0], -1.0 as $ty);
            assert_eq!(ddata[0], 0.0 as $ty);

            let mut dout = vec![-10.0 as $ty; n_outputs];
            compiler.calc_out_grad(0.0 as $ty, &u, &du, &data, &mut ddata, &out, &mut dout);
            assert_eq!(dout[0], 1.0 as $ty);
            assert_eq!(ddata[0], 0.0 as $ty);

            let mut dinputs = vec![1.0 as $ty; n_inputs];
            compiler.set_inputs_grad(&inputs, &dinputs, &data, &mut ddata, 0);
            assert_eq!(ddata[0], 1.0 as $ty);

            let mut du_rev = vec![-11.0 as $ty; n_states];
            let mut ddata_rev = vec![-12.0 as $ty; n_data];
            let mut drr_rev = vec![1.0 as $ty; n_states];
            compiler.rhs_rgrad(
                0.0 as $ty,
                &u,
                &mut du_rev,
                &data,
                &mut ddata_rev,
                &rr,
                &mut drr_rev,
            );
            assert_eq!(du_rev[0], -12.0 as $ty);
            assert_eq!(ddata_rev[0], -12.0 as $ty);

            let mut dv = vec![-13.0 as $ty; n_states];
            let mut dmv = vec![1.0 as $ty; n_states];
            compiler.mass_rgrad(0.0 as $ty, &mut dv, &data, &mut ddata_rev, &mut dmv);
            assert_eq!(dv[0], -12.0 as $ty);

            let mut dout_rev = vec![1.0 as $ty; n_outputs];
            compiler.calc_out_rgrad(
                0.0 as $ty,
                &u,
                &mut du_rev,
                &data,
                &mut ddata_rev,
                &out,
                &mut dout_rev,
            );
            assert_eq!(du_rev[0], -11.0 as $ty);

            compiler.set_inputs_rgrad(&inputs, &mut dinputs, &data, &mut ddata_rev, 0);
            assert_eq!(dinputs[0], -11.0 as $ty);

            let mut ddata_s = vec![-14.0 as $ty; n_data];
            let mut drr_s = vec![-15.0 as $ty; n_states];
            compiler.rhs_sgrad(0.0 as $ty, &u, &data, &mut ddata_s, &rr, &mut drr_s);
            assert_eq!(drr_s[0], 0.0 as $ty);
            assert_eq!(ddata_s[0], 0.0 as $ty);

            let mut dout_s = vec![-16.0 as $ty; n_outputs];
            compiler.calc_out_sgrad(0.0 as $ty, &u, &data, &mut ddata_s, &out, &mut dout_s);
            assert_eq!(dout_s[0], 0.0 as $ty);

            let mut ddata_sr = vec![-17.0 as $ty; n_data];
            let mut drr_sr = vec![-18.0 as $ty; n_states];
            compiler.rhs_srgrad(0.0 as $ty, &u, &data, &mut ddata_sr, &rr, &mut drr_sr);
            assert_eq!(drr_sr[0], 0.0 as $ty);
            assert_eq!(ddata_sr[0], 0.0 as $ty);

            let mut dout_sr = vec![-19.0 as $ty; n_outputs];
            compiler.calc_out_srgrad(0.0 as $ty, &u, &data, &mut ddata_sr, &out, &mut dout_sr);
            assert_eq!(dout_sr[0], 0.0 as $ty);
        }
    };
}
