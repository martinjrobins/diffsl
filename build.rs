use std::env;

fn compile_enzyme(llvm_dir: String) -> String {
    let dst = cmake::Config::new("Enzyme/enzyme")
        .define("LLVM_DIR", llvm_dir)
        .build();
    let dst_disp = dst.display();
    let lib_loc = format!("{}/lib", dst_disp);
    lib_loc
}

fn main() {
    // get env vars matching DEP_LLVM_*_LIBDIR regex    
    let llvm_dirs: Vec<_> = env::vars().filter(|(k, _)| k.starts_with("DEP_LLVM_") && k.ends_with("_LIBDIR")).collect();
    // take first one
    let llvm_dir = llvm_dirs.first().expect("DEP_LLVM_*_LIBDIR not set").1.clone();

    dbg!("llvm_dir", &llvm_dir);
    
    // compile enzyme
    let libdir= compile_enzyme(llvm_dir);
    println!("cargo:rustc-link-search=native={}", libdir);
    println!("cargo:rerun-if-changed=build.rs");
}