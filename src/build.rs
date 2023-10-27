use std::env;
use std::path::PathBuf;

fn generate_enzyme_wrapper() {

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", std::env::var("ENZYME_LIB_DIR").unwrap_or("/usr/local/lib".to_string()));

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=LLVMEnzyme-14");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/enzyme/wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/enzyme/wrapper.h")
        // set the clang args to include the llvm prefix dir
        .clang_arg(format!("-I{}", std::env::var("LLVM_SYS_14_PREFIX/include").unwrap_or("/usr/lib/llvm-14/include".to_string())))
        // set the clang args to include the llvm include dir
        .clang_arg(format!("-I{}", std::env::var("ENZYME_INCLUDE_DIR").unwrap_or("/usr/local/include".to_string())))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    generate_enzyme_wrapper();
}