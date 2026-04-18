#[cfg(all(
    feature = "llvm",
    feature = "external_dynamic",
    not(target_arch = "wasm32")
))]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use diffsl::discretise::DiscreteModel;
    use diffsl::execution::compiler::{CompilerMode, CompilerOptions};
    use diffsl::execution::module::CodegenModuleCompile;
    use diffsl::execution::scalar::RealType;
    use diffsl::parser::parse_ds_string;
    use diffsl::{Compiler, ExternalDynModule, LlvmModule};

    fn dynamic_library_name(stem: &str) -> String {
        if cfg!(target_os = "windows") {
            format!("{stem}.dll")
        } else if cfg!(target_os = "macos") {
            format!("lib{stem}.dylib")
        } else {
            format!("lib{stem}.so")
        }
    }

    #[test]
    fn llvm_module_dynamic_library_roundtrip_rhs() {
        let code = r#"
            u { y = 1 }
            F { -y }
            out { y }
        "#;

        let ast = parse_ds_string(code).expect("dsl should parse");
        let model = DiscreteModel::build("llvm_dynamic_roundtrip", &ast)
            .expect("discrete model should build");

        let llvm_module = <LlvmModule as CodegenModuleCompile>::from_discrete_model(
            &model,
            CompilerOptions::default(),
            None,
            RealType::F64,
            Some(code),
        )
        .expect("llvm module should compile");

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let lib_name = dynamic_library_name(&format!("diffsl_llvm_roundtrip_{stamp}"));
        let output_path = std::env::temp_dir().join(lib_name);

        llvm_module
            .to_dynamic_library(output_path.clone())
            .expect("dynamic library should be written");

        let dyn_module =
            ExternalDynModule::<f64>::new(&output_path).expect("dynamic library should load");
        let compiler = Compiler::from_codegen_module(dyn_module, CompilerMode::SingleThreaded)
            .expect("compiler should build");

        let mut data = compiler.get_new_data();
        let mut u = vec![0.0_f64; 1];
        compiler.set_u0(&mut u, &mut data);

        let mut rr = vec![0.0_f64; 1];
        compiler.rhs(0.0_f64, &u, &mut data, &mut rr);

        assert_eq!(u[0], 1.0_f64);
        assert_eq!(rr[0], -1.0_f64);

        let _ = std::fs::remove_file(output_path);
    }
}
