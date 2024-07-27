use bindgen::{BindgenError, Bindings, Builder};
use std::{env, path::PathBuf};

fn compile_enzyme(llvm_dir: String) -> (String, String) {
    let dst = cmake::Config::new("Enzyme/enzyme")
        .define("ENZYME_STATIC_LIB", "ON")
        .define("ENZYME_CLANG", "OFF")
        .define("LLVM_DIR", llvm_dir)
        .define(
            "CMAKE_CXX_FLAGS",
            "-Wno-comment -Wno-deprecated-declarations",
        )
        .build();
    let dst_disp = dst.display();
    let lib_dir = format!("{}/lib", dst_disp);
    let inc_dir = "Enzyme/enzyme".to_string();
    (lib_dir, inc_dir)
}

fn enzyme_bindings(inc_dirs: &[String]) -> Result<Bindings, BindgenError> {
    dbg!(inc_dirs);
    let mut builder = Builder::default()
        .header("wrapper.h")
        .generate_comments(false)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-fvisibility=default")
        .allowlist_function("CreateEnzymeLogic")
        .allowlist_function("FreeEnzymeLogic")
        .allowlist_function("CreateTypeAnalysis")
        .allowlist_function("FreeTypeAnalysis")
        .allowlist_function("EnzymeNewTypeTreeCT")
        .allowlist_function("EnzymeFreeTypeTree")
        .allowlist_function("EnzymeMergeTypeTree")
        .allowlist_function("EnzymeTypeTreeOnlyEq")
        .allowlist_function("EnzymeCreateForwardDiff")
        ;

    // add include dirs
    for dir in inc_dirs {
        builder = builder.clang_arg(format!("-I{}", dir))
    }
    builder.generate()
}

fn main() {
    // Get llvm directory from llvm-sys dependency
    // get env vars matching DEP_LLVM_*_LIBDIR regex
    let llvm_dirs: Vec<_> = env::vars()
        .filter(|(k, _)| k.starts_with("DEP_LLVM_") && k.ends_with("_LIBDIR"))
        .collect();
    // take first one
    let llvm_lib_dir = llvm_dirs
        .first()
        .expect("DEP_LLVM_*_LIBDIR not set")
        .1
        .clone();
    let llvm_env_key = llvm_dirs.first().unwrap().0.clone();
    let llvm_version = &llvm_env_key["DEP_LLVM_".len()..(llvm_env_key.len() - "_LIBDIR".len())];
    dbg!(llvm_version);

    // replace last "lib" with "include"
    let llvm_inc_dir = llvm_lib_dir
        .chars()
        .take(llvm_lib_dir.len() - 3)
        .collect::<String>()
        + "include";

    // compile enzyme or use provided paths
    let (libdir, incdir) = if cfg!(feature = "no-build-enzyme") {
        let incdir = env::var("ENZYME_INCLUDE_DIR").ok();
        let libdir = env::var("ENZYME_LIBRARY_DIR").ok();
        if let (Some(incdir), Some(libdir)) = (incdir, libdir) {
            (libdir, incdir)
        } else {
            panic!("environment variables ENZYME_INCLUDE_DIR and ENZYME_LIBRARY_DIR must be set if not building enzyme");
        }
    } else {
        compile_enzyme(llvm_lib_dir.clone())
    };

    let libnames = [format!("EnzymeStatic-{}", llvm_version)];

    // bind enzyme api
    let bindings_rs = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    let bindings = enzyme_bindings(&[llvm_inc_dir, incdir]).expect("Couldn't generate bindings!");
    bindings
        .write_to_file(bindings_rs)
        .expect("Couldn't write file bindings.rs!");

    println!("cargo:rustc-link-search=native={}", libdir);
    println!("cargo:rustc-link-search=native={}", llvm_lib_dir);
    for libname in libnames.iter() {
        println!("cargo:rustc-link-lib={}", libname);
    }
    println!("cargo:rustc-link-lib=LLVMDemangle");
    println!("cargo:rerun-if-changed=wrapper.h");
}
