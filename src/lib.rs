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


pub struct CompilerOptions {
    pub bitcode_only: bool,
    pub wasm: bool,
    pub standalone: bool,
}

fn is_executable_on_path(executable_name: &str) -> bool {
    let output = Command::new("which")
        .arg(executable_name)
        .output()
        .expect("failed to execute which command");

    output.status.success()
}

fn find_executable<'a>(varients: &[&'a str]) -> Result<&'a str> {
    let mut command = None;
    for varient in varients {
        if is_executable_on_path(varient) {
            command = Some(varient.to_owned());
            break;
        }
    }
    match command {
        Some(command) => Ok(command),
        None => Err(anyhow!("Could not find any of {:?} on path", varients)),
    }
}


fn find_runtime_path(libraries: &[&str] ) -> Result<String> {
    let library_paths_env = env::var("LIBRARY_PATH").unwrap_or("".to_owned());
    let library_paths = library_paths_env.split(":").collect::<Vec<_>>();
    for path in library_paths {
        // check if all librarys are in the path
        let mut found = true;
        for library in libraries {
            let library_path = Path::new(path).join(library);
            if !library_path.exists() {
                found = false;
                break;
            }
        }
        if found {
            return Ok(path.to_owned());
        }
    }
    Err(anyhow!("Could not find {:?} in LIBRARY_PATH", libraries))
}


pub fn compile(input: &str, out: Option<&str>, model: Option<&str>, options: CompilerOptions) -> Result<()> {
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
    let out = if let Some(out) = out {
        out.clone()
    } else if options.bitcode_only {
        "out.ll"
    } else {
        "out"
    };
    let text = std::fs::read_to_string(inputfile)?;
    compile_text(text.as_str(), out, model_name, options, is_discrete)
}

pub fn compile_text(text: &str, out: &str, model_name: &str, options: CompilerOptions, is_discrete: bool) -> Result<()> {
    let CompilerOptions { bitcode_only, wasm, standalone } = options;
    
    let is_continuous = !is_discrete;

    let bitcodename = if bitcode_only { out.to_owned() } else { format!("{}.bc", out) };
    let bitcodefile = Path::new(bitcodename.as_str());
    let pre_enzyme_bitcodefile = bitcodefile.with_extension("pre_enzyme.bc");
    
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
            let error_text = model_info.errors.iter().fold(String::new(), |acc, error| {
                format!("{}\n{}", acc, error.as_error_message(text))
            });
            return Err(anyhow!(error_text));
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
                return Err(anyhow!(e.as_error_message(text)));
            }
        }
    } else {
        panic!("No model found");
    };

    let compiler = Compiler::from_discrete_model(&discrete_model)?;

    compiler.write_bitcode_to_path(&pre_enzyme_bitcodefile)?;
    
    let opt_name_varients = ["opt-14"];
    let opt_name = find_executable(&opt_name_varients)?;
    let enzyme_lib_path = find_runtime_path(&["LLVMEnzyme-14.so"])?;
    let enzyme_lib = Path::new(enzyme_lib_path.as_str()).join("LLVMEnzyme-14.so");

    let output = Command::new(opt_name)
        .arg(pre_enzyme_bitcodefile.to_str().unwrap())
        .arg(format!("-load={}", enzyme_lib.to_str().unwrap()))
        .arg("-enzyme")
        .arg("--enable-new-pm=0")
        .arg("-o").arg(bitcodefile.to_str().unwrap())
        .arg("-S")
        .output()?;
    
    if let Some(code) = output.status.code() {
        if code != 0 {
            println!("{}", String::from_utf8_lossy(&output.stderr));
            return Err(anyhow!("{} returned error code {}", opt_name, code));
        }
    }
    
    // if we are only compiling to bitcode , we are done
    if bitcode_only {
        return Ok(());
    }
    
    // compile the bitcode to an object file or standalone wasm or executable
    let emcc_varients = ["emcc"];
    let clang_varients = ["clang", "clang-14"];
    let command_name = if wasm { 
        find_executable(&emcc_varients)?
    } else { 
        find_executable(&clang_varients)?
    };
    
    
    // link the object file and our runtime library
    let output = if wasm {
        let exported_functions = vec![
            "Vector_destroy",
            "Vector_create",
            "Vector_create_with_capacity",
            "Vector_push",
            
            "Options_destroy",
            "Options_create",

            "Sundials_destroy",
            "Sundials_create",
            "Sundials_init",
            "Sundials_solve",
        ];
        let mut linked_files = vec![
            "libdiffeq_runtime_lib_wasm.a",
            "libsundials_idas_wasm.a",
            "libargparse_wasm.a",
        ];
        if standalone {
            linked_files.push("libdiffeq_runtime_wasm.a");
        }
        let linked_files = linked_files;
        let runtime_path = find_runtime_path(&linked_files)?;
        let mut command = Command::new(command_name);
        command.arg("-o").arg(out).arg(out);
        for file in linked_files {
            command.arg(Path::new(runtime_path.as_str()).join(file));
        }
        if !standalone {
            let exported_functions = exported_functions.into_iter().map(|s| format!("_{}", s)).collect::<Vec<_>>().join(",");
            command.arg("-s").arg(format!("EXPORTED_FUNCTIONS={}", exported_functions));
            command.arg("--no-entry");
        }
        command.output()
    } else {
        let mut command = Command::new(command_name);
        command.arg("-o").arg(out).arg(out);
        if standalone {
            command.arg("-ldiffeq_runtime");
        } else {
            command.arg("-ldiffeq_runtime_lib");
        }
        command.output()
    };

    let output = match output {
        Ok(output) => output,
        Err(e) => {
            return Err(anyhow!("Error running {}: {}", command_name, e));
        }
    };
    
    if let Some(code) = output.status.code() {
        if code != 0 {
            println!("{}", String::from_utf8_lossy(&output.stderr));
            return Err(anyhow!("{} returned error code {}", command_name, code));
        }
    }

    Ok(())
}




