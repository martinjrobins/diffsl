#[cfg(any(
    all(feature = "llvm", not(target_arch = "wasm32")),
    all(
        feature = "cranelift",
        not(target_arch = "wasm32"),
        not(target_os = "macos")
    )
))]
fn model_code() -> &'static str {
    r#"
        u { y = 1 }
        F { -y }
        out { y }
    "#
}

#[cfg(any(
    all(feature = "llvm", not(target_arch = "wasm32")),
    all(
        feature = "cranelift",
        not(target_arch = "wasm32"),
        not(target_os = "macos")
    )
))]
fn unique_stamp() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_nanos()
}

#[cfg(any(
    all(feature = "llvm", not(target_arch = "wasm32")),
    all(
        feature = "cranelift",
        not(target_arch = "wasm32"),
        not(target_os = "macos")
    )
))]
fn assert_rhs_works<M: diffsl::execution::module::CodegenModule>(
    compiler: &diffsl::Compiler<M, f64>,
) {
    let mut data = compiler.get_new_data();
    let mut u = vec![0.0_f64; 1];
    compiler.set_u0(&mut u, &mut data);

    let mut rr = vec![0.0_f64; 1];
    compiler.rhs(0.0_f64, &u, &mut data, &mut rr);

    assert_eq!(u[0], 1.0_f64);
    assert_eq!(rr[0], -1.0_f64);
}

#[cfg(all(
    feature = "llvm",
    feature = "external_dynamic",
    not(target_arch = "wasm32")
))]
#[test]
fn llvm_dynamic_library_roundtrip() {
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

    let code = model_code();
    let ast = parse_ds_string(code).expect("dsl should parse");
    let model =
        DiscreteModel::build("llvm_dynamic_roundtrip", &ast).expect("discrete model should build");

    let llvm_module = <LlvmModule as CodegenModuleCompile>::from_discrete_model(
        &model,
        CompilerOptions::default(),
        None,
        RealType::F64,
        Some(code),
    )
    .expect("llvm module should compile");

    let lib_name = dynamic_library_name(&format!("diffsl_llvm_roundtrip_{}", unique_stamp()));
    let output_path = std::env::temp_dir().join(lib_name);

    llvm_module
        .to_dynamic_library(output_path.clone())
        .expect("dynamic library should be written");

    let dyn_module =
        ExternalDynModule::<f64>::new(&output_path).expect("dynamic library should load");
    let compiler = Compiler::from_codegen_module(dyn_module, CompilerMode::SingleThreaded)
        .expect("compiler should build");

    assert_rhs_works(&compiler);
    let _ = std::fs::remove_file(output_path);
}

#[cfg(all(feature = "llvm", not(target_arch = "wasm32")))]
#[test]
fn llvm_module_object_file_roundtrip() {
    use diffsl::discretise::DiscreteModel;
    use diffsl::execution::compiler::{CompilerMode, CompilerOptions};
    use diffsl::execution::module::{CodegenModuleCompile, CodegenModuleEmit};
    use diffsl::execution::scalar::RealType;
    use diffsl::parser::parse_ds_string;
    use diffsl::{Compiler, LlvmModule, ObjectModule};

    let code = model_code();
    let ast = parse_ds_string(code).expect("dsl should parse");
    let model =
        DiscreteModel::build("llvm_object_roundtrip", &ast).expect("discrete model should build");

    let llvm_module = <LlvmModule as CodegenModuleCompile>::from_discrete_model(
        &model,
        CompilerOptions::default(),
        None,
        RealType::F64,
        Some(code),
    )
    .expect("llvm module should compile");

    let object_path =
        std::env::temp_dir().join(format!("diffsl_llvm_roundtrip_{}.o", unique_stamp()));

    let object_buffer = llvm_module
        .to_object()
        .expect("object file should be generated");
    std::fs::write(&object_path, &object_buffer).expect("object file should be written");

    let object_buffer = std::fs::read(&object_path).expect("object file should be readable");
    let compiler = Compiler::<ObjectModule, f64>::from_object_file(
        object_buffer,
        CompilerMode::SingleThreaded,
    )
    .expect("compiler should build from object file");

    assert_rhs_works(&compiler);
    let _ = std::fs::remove_file(object_path);
}

#[cfg(all(
    feature = "cranelift",
    not(target_arch = "wasm32"),
    not(target_os = "macos")
))]
#[test]
fn cranelift_module_object_file_roundtrip() {
    use diffsl::discretise::DiscreteModel;
    use diffsl::execution::compiler::{CompilerMode, CompilerOptions};
    use diffsl::execution::module::{CodegenModuleCompile, CodegenModuleEmit};
    use diffsl::execution::scalar::RealType;
    use diffsl::parser::parse_ds_string;
    use diffsl::{Compiler, CraneliftObjectModule, ObjectModule};

    let code = model_code();
    let ast = parse_ds_string(code).expect("dsl should parse");
    let model = DiscreteModel::build("cranelift_object_roundtrip", &ast)
        .expect("discrete model should build");

    let cranelift_module = <CraneliftObjectModule as CodegenModuleCompile>::from_discrete_model(
        &model,
        CompilerOptions::default(),
        None,
        RealType::F64,
        Some(code),
    )
    .expect("cranelift module should compile");

    let object_path =
        std::env::temp_dir().join(format!("diffsl_cranelift_roundtrip_{}.o", unique_stamp()));

    let object_buffer = cranelift_module
        .to_object()
        .expect("object file should be generated");
    std::fs::write(&object_path, &object_buffer).expect("object file should be written");

    let object_buffer = std::fs::read(&object_path).expect("object file should be readable");
    let compiler = Compiler::<ObjectModule, f64>::from_object_file(
        object_buffer,
        CompilerMode::SingleThreaded,
    )
    .expect("compiler should build from object file");

    assert_rhs_works(&compiler);
    let _ = std::fs::remove_file(object_path);
}

#[cfg(all(
    feature = "cranelift",
    not(target_arch = "wasm32"),
    not(target_os = "macos")
))]
#[test]
fn cranelift_multiple_to_object_calls_roundtrip() {
    use diffsl::discretise::DiscreteModel;
    use diffsl::execution::compiler::{CompilerMode, CompilerOptions};
    use diffsl::execution::module::{CodegenModuleCompile, CodegenModuleEmit};
    use diffsl::execution::scalar::RealType;
    use diffsl::parser::parse_ds_string;
    use diffsl::{Compiler, CraneliftObjectModule, ObjectModule};

    let code = model_code();
    let ast = parse_ds_string(code).expect("dsl should parse");
    let model = DiscreteModel::build("cranelift_multiple_object_roundtrip", &ast)
        .expect("discrete model should build");

    let cranelift_module = <CraneliftObjectModule as CodegenModuleCompile>::from_discrete_model(
        &model,
        CompilerOptions::default(),
        None,
        RealType::F64,
        Some(code),
    )
    .expect("cranelift module should compile");

    let object_buffer_1 = cranelift_module
        .to_object()
        .expect("first object emission should succeed");
    let object_buffer_2 = cranelift_module
        .to_object()
        .expect("second object emission should succeed");

    assert!(!object_buffer_1.is_empty());
    assert_eq!(object_buffer_1, object_buffer_2);

    let compiler = Compiler::<ObjectModule, f64>::from_object_file(
        object_buffer_2,
        CompilerMode::SingleThreaded,
    )
    .expect("compiler should build from object file");

    assert_rhs_works(&compiler);
}
