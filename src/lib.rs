use std::{path::Path, ffi::OsStr, process::Command, env};
use anyhow::{Result, anyhow};
use codegen::Compiler;
use continuous::ModelInfo;
use discretise::DiscreteModel;
use parser::{parse_ms_string, parse_ds_string};

extern crate pest;
#[macro_use]
extern crate pest_derive;

pub mod parser;
pub mod ast;
pub mod discretise;
pub mod continuous;
pub mod codegen;


pub fn compile(input: &str, out: Option<&str>, model: Option<&str>, compile: bool, wasm: bool) -> Result<()> {
    let inputfile = Path::new(input);
    let is_discrete = inputfile.extension().unwrap_or(OsStr::new("")).to_str().unwrap() == "ds";
    let is_continuous = inputfile.extension().unwrap_or(OsStr::new("")).to_str().unwrap() == "cs";
    if !is_discrete && !is_continuous {
        panic!("Input file must have extension .ds or .cs");
    }
    let model_name = if is_continuous {
        if let Some(model_name) = model {
            model_name
        } else {
            return Err(anyhow!("Model name must be specified for continuous models"));
        }
    } else {
        inputfile.file_stem().unwrap().to_str().unwrap()
    };
    let text = std::fs::read_to_string(inputfile)?;
    compile_text(text.as_str(), out, model_name, compile, wasm, is_discrete)
}

pub fn compile_text(text: &str, out: Option<&str>, model_name: &str, compile: bool, wasm: bool, is_discrete: bool) -> Result<()> {
    let out = if let Some(out) = out {
        out.clone()
    } else if compile {
        "out.o"
    } else {
        "out"
    };
    let is_continuous = !is_discrete;

    let objectname = if compile { out.to_owned() } else { format!("{}.o", out) };
    let objectfile = Path::new(objectname.as_str());
    
    let continuous_ast = if is_continuous {
        Some(parse_ms_string(text)?)
    } else {
        None
    };

    let discrete_ast = if is_discrete {
        Some(parse_ds_string(text)?)
    } else {
        None
    }; 

    let continuous_model_info = if let Some(ast) = &continuous_ast {
        let model_info = ModelInfo::build(model_name, ast).map_err(|e| anyhow!("{}", e))?;
        if model_info.errors.len() > 0 {
            for error in model_info.errors {
                println!("{}", error.as_error_message(text));
            }
            return Err(anyhow!("Errors in model"));
        }
        Some(model_info)
    } else {
        None
    };

    let discrete_model = if let Some(model_info) = &continuous_model_info {
        let model = DiscreteModel::from(&model_info);
        model
    } else if let Some(ast) = &discrete_ast {
        match DiscreteModel::build(model_name, ast) {
            Ok(model) => model,
            Err(e) => {
                println!("{}", e.as_error_message(text));
                return Err(anyhow!("Errors in model"));
            }
        }
    } else {
        panic!("No model found");
    };

    let compiler = Compiler::from_discrete_model(&discrete_model)?;

    if wasm {
        compiler.write_wasm_object_file(objectfile)?;
    } else {
        compiler.write_object_file(objectfile)?;
    }
    
    if compile {
        return Ok(());
    }
    
    
    // link the object file and our runtime library
    let output = if wasm {
        let library_paths_env = env::var("LIBRARY_PATH").unwrap_or("".to_owned());
        let library_paths = library_paths_env.split(":").collect::<Vec<_>>();
        let mut runtime_path = None;
        for path in library_paths {
            // check if the library path contains the runtime library
            let runtime_path_test = Path::new(path).join("libdiffeq_runtime_wasm.a");
            if runtime_path_test.exists() {
                runtime_path = Some(path);
            }
        };
        if runtime_path.is_none() {
            return Err(anyhow!("Could not find libdiffeq_runtime_wasm.a in LIBRARY_PATH"));
        }
        let runtime_path = runtime_path.unwrap();
        let runtime_file = Path::new(runtime_path).join("libdiffeq_runtime_wasm.a");
        let runtime_lib_file = Path::new(runtime_path).join("libdiffeq_runtime_lib_wasm.a");
        let sundials_lib_file = Path::new(runtime_path).join("libsundials_idas_wasm.a");
        let argparse_lib_file = Path::new(runtime_path).join("libargparse_wasm.a");
        Command::new("emcc")
            .arg("-o")
            .arg(out)
            .arg(objectname.clone())
            .arg(runtime_file)
            .arg(runtime_lib_file)
            .arg(sundials_lib_file)
            .arg(argparse_lib_file)
            .output()?
    } else {
        Command::new("clang")
            .arg("-o")
            .arg(out)
            .arg(objectname.clone())
            .arg("-ldiffeq_runtime")
            .output()?      
    };
    
    
    // clean up the object file
    std::fs::remove_file(objectfile)?;
    
    if let Some(code) = output.status.code() {
        if code != 0 {
            println!("{}", String::from_utf8_lossy(&output.stderr));
            return Err(anyhow!("clang returned error code {}", code));
        }
    }
    Ok(())
}




