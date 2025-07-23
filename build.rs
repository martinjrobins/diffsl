#[cfg(feature = "enzyme")]
mod enzyme {
    use bindgen::{BindgenError, Bindings, Builder};
    use regex::Regex;
    use std::{env, io::{self, stdout}, path::PathBuf, process::Command};
    use std::io::Write;

    fn get_llvm_static_libs(llvm_lib_dir: String) -> Vec<String> {
        // get all the static libraries from the LLVM lib dir
        //let llvm_lib_dir = llvm_lib_dir.trim_end_matches('/'); // ensure no trailing slash
        //if !std::path::Path::new(&llvm_lib_dir).exists() {
        //    panic!("LLVM lib directory does not exist: {llvm_lib_dir}");
        //}
        //let mut libs = Vec::new();
        //for entry in std::fs::read_dir(llvm_lib_dir).expect("Could not read LLVM lib directory") {
        //    let entry = entry.expect("Could not read entry in LLVM lib directory");
        //    if entry.path().extension().is_some_and(|ext| ext == "a" || ext == "lib") {
        //        // only include static libraries
        //        libs.push(entry.file_name().to_string_lossy().into_owned());
        //    }
        //}
        //// remove the prefix "lib" and suffix ".a" or ".lib" from the library names
        //libs
        //    .into_iter()
        //    .map(|lib| lib.replace("lib", "").replace(".a", "").replace(".lib", ""))
        //    .collect::<Vec<_>>()
        let llvm_cmake_dir = format!("{llvm_lib_dir}/cmake/llvm");
        // read LLVMConfig.cmake file and extract LLVM_AVAILABLE_LIBS from the line
        // that starts with set(LLVM_AVAILABLE_LIBS and ends with )
        //let reg = Regex::new(r#"set\(LLVM_AVAILABLE_LIBS\s+([^)]+)\)"#).unwrap();

        //let file = std::fs::read_to_string(format!("{llvm_cmake_dir}/LLVMConfig.cmake"))
        //    .expect("Could not read LLVMConfig.cmake file");
        //let search_result = reg
        //    .captures(&file)
        //    .and_then(|cap| cap.get(1))
        //    .map_or_else(
        //        || {
        //            panic!("Could not find LLVM_AVAILABLE_LIBS in LLVMConfig.cmake file");
        //        },
        //        |m| m.as_str(),
        //    );
        //search_result 
        //    .split(";")
        //    .filter(|s| s.starts_with("LLVM") && s != &"LLVM")
        //    .map(|s| s.to_string())
        //    .collect()
        let output = Command::new("cmake")
            .arg(format!("-DCMAKE_PREFIX_PATH={llvm_lib_dir}"))
            .arg(".")
            .output()
            .expect("Failed to run cmake command");
        io::stdout().write_all(&output.stdout).unwrap();
        if !output.status.success() {
            io::stderr().write_all(&output.stderr).unwrap();
            panic!("CMake command failed");
        }
        let output = String::from_utf8(output.stdout).unwrap();
        // extract pattern _START_<libs> _END_ from the output
        let re = Regex::new(r"_START_(.*?)_END_").unwrap();
        let caps = re.captures(&output).expect("Could not find _START_ and _END_ in output");
        let libs_str = caps.get(1).expect("Could not find libs in output").as_str();
        // split the libs_str by semicolon and trim
        libs_str.split(";").map(|s| s.trim().to_string()).collect()
    }

    // taken from https://gitlab.com/taricorp/llvm-sys.rs/-/blob/main/build.rs
    // MIT License
    fn target_env_is(name: &str) -> bool {
        match env::var_os("CARGO_CFG_TARGET_ENV") {
            Some(s) => s == name,
            None => false,
        }
    }

    // taken from https://gitlab.com/taricorp/llvm-sys.rs/-/blob/main/build.rs
    // MIT License
    fn target_os_is(name: &str) -> bool {
        match env::var_os("CARGO_CFG_TARGET_OS") {
            Some(s) => s == name,
            None => false,
        }
    }

    // taken from https://gitlab.com/taricorp/llvm-sys.rs/-/blob/main/build.rs
    // MIT License
    /// Get the library that must be linked for C++, if any.
    fn get_system_libcpp() -> Option<&'static str> {
        if let Some(libcpp) = option_env!("LLVM_SYS_LIBCPP") {
            // Use the library defined by the caller, if provided.
            Some(libcpp)
        } else if target_env_is("msvc") {
            // MSVC doesn't need an explicit one.
            None
        } else if target_os_is("macos") {
            // On OS X 10.9 and later, LLVM's libc++ is the default. On earlier
            // releases GCC's libstdc++ is default. Unfortunately we can't
            // reasonably detect which one we need (on older ones libc++ is
            // available and can be selected with -stdlib=lib++), so assume the
            // latest, at the cost of breaking the build on older OS releases
            // when LLVM was built against libstdc++.
            Some("c++")
        } else if target_os_is("freebsd") || target_os_is("openbsd") {
            Some("c++")
        } else {
            // Otherwise assume GCC's libstdc++.
            // This assumption is probably wrong on some platforms, but it can be
            // always overwritten through `LLVM_SYS_LIBCPP` variable.
            Some("stdc++")
        }
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
        for lib in llvm_libs {
            println!("cargo:rustc-link-lib=static={lib}");
        }
        if let Some(libcpp) = get_system_libcpp() {
            println!("cargo:rustc-link-lib=dylib={libcpp}");
        }
        //println!("cargo:rustc-link-lib=LLVMDemangle");
        println!("cargo:rerun-if-changed=wrapper.h");

    }
}

fn main() {
    #[cfg(feature = "enzyme")]
    enzyme::enzyme_main();
}
