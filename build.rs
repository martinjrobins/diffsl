#[cfg(feature = "enzyme")]
mod enzyme {
    use bindgen::{BindgenError, Bindings, Builder};
    use regex::Regex;
    use std::{env, path::PathBuf, process::Command};

    fn get_llvm_static_libs(llvm_lib_dir: String) -> Vec<String> {
        let llvm_cmake_dir = format!("{llvm_lib_dir}/cmake/llvm");
        // read LLVMConfig.cmake file and extract LLVM_AVAILABLE_LIBS from the line
        // that starts with set(LLVM_AVAILABLE_LIBS and ends with )
        let reg = Regex::new(r#"set\(LLVM_AVAILABLE_LIBS\s+([^)]+)\)"#).unwrap();

        let file = std::fs::read_to_string(format!("{llvm_cmake_dir}/LLVMConfig.cmake"))
            .expect("Could not read LLVMConfig.cmake file");
        let search_result = reg
            .captures(&file)
            .and_then(|cap| cap.get(1))
            .map_or_else(
                || {
                    panic!("Could not find LLVM_AVAILABLE_LIBS in LLVMConfig.cmake file");
                },
                |m| m.as_str(),
            );
        search_result 
            .split(";")
            .map(|s| s.to_string())
            .collect()
    }

    fn compile_enzyme(llvm_lib_dir: String) -> (String, String) {
        let llvm_cmake_dir = format!("{llvm_lib_dir}/cmake/llvm");
        let dst = cmake::Config::new("Enzyme/enzyme")
            .define("ENZYME_STATIC_LIB", "ON")
            .define("ENZYME_CLANG", "OFF")
            .define("LLVM_DIR", llvm_cmake_dir)
            .define(
                "CMAKE_CXX_FLAGS",
                "-Wno-comment -Wno-deprecated-declarations",
            )
            .build();
        let out_dir = dst.display().to_string();
        let inc_dir = "Enzyme/enzyme".to_string();
        (out_dir, inc_dir)
    }

    fn enzyme_bindings(inc_dirs: &[String]) -> Result<Bindings, BindgenError> {
        let mut builder = Builder::default()
            .header("wrapper.h")
            .generate_comments(false)
            .blocklist_type("LLVMBuilderRef")
            .blocklist_type("LLVMValueRef")
            .clang_arg("-x")
            .clang_arg("c++");

        // add include dirs
        for dir in inc_dirs {
            builder = builder.clang_arg(format!("-I{dir}"))
        }
        if cfg!(target_os = "macos") {
            let xcode_inc_dir = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include";
            builder = builder.clang_arg(format!("-I{xcode_inc_dir}"));
        }

        builder.generate()
    }

    pub fn enzyme_main() {
        let llvm_version = if cfg!(any(feature = "llvm15-0", feature = "llvm15-0-manual")) {
            "15"
        } else if cfg!(any(feature = "llvm16-0", feature = "llvm16-0-manual")) {
            "16"
        } else if cfg!(any(feature = "llvm17-0", feature = "llvm17-0-manual")) {
            "17"
        } else if cfg!(any(feature = "llvm18-1", feature = "llvm18-1-manual")) {
            "18"
        } else {
            panic!("No LLVM version feature enabled");
        };
        dbg!(llvm_version);
        let llvm_lib_dir = if cfg!(feature = "llvm-manual") {
            // get dir from LLVM_DIR env var
            let llvm_dir = env::var("LLVM_DIR").expect("LLVM_DIR env var not set");
            format!("{llvm_dir}/lib")
        } else {
            // get env vars matching DEP_LLVM_*_LIBDIR regex from llvm-sys
            env::vars()
            .filter(|(k, _)| k.starts_with("DEP_LLVM_") && k.ends_with("_LIBDIR"))
            .collect::<Vec<_>>()
            .first()
            .expect("DEP_LLVM_*_LIBDIR not set")
            .1
            .clone()
        };

        // replace last "lib" with "include"
        let llvm_inc_dir = llvm_lib_dir
            .chars()
            .take(llvm_lib_dir.len() - 3)
            .collect::<String>()
            + "include";

        // compile enzyme
        let (outdir, incdir) = compile_enzyme(llvm_lib_dir.clone());
        let libnames = [format!("EnzymeStatic-{llvm_version}")];

        // bind enzyme api
        let bindings_rs = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
        let bindings =
            enzyme_bindings(&[llvm_inc_dir, incdir]).expect("Couldn't generate bindings!");
        bindings
            .write_to_file(bindings_rs)
            .expect("Couldn't write file bindings.rs!");

        println!("cargo:rustc-link-search=native={llvm_lib_dir}");
        let llvm_libs = if cfg!(feature = "llvm-manual") {
            get_llvm_static_libs(llvm_lib_dir)
        } else {
            vec![]
        };
        for lib in llvm_libs {
            println!("cargo:rustc-link-lib=static={lib}");
        }

        for libname in ["lib", "lib64", "lib32"] {
            let libdir = format!("{outdir}/{libname}");
            println!("cargo:rustc-link-search=native={libdir}");
        }
        // add homebrew lib dir if on macos, needed for zstd libraries
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        }
        for libname in libnames.iter() {
            println!("cargo:rustc-link-lib={libname}");
        }
        println!("cargo:rustc-link-lib=LLVMDemangle");
        println!("cargo:rerun-if-changed=wrapper.h");
    }
}

fn main() {
    #[cfg(feature = "enzyme")]
    enzyme::enzyme_main();
}
