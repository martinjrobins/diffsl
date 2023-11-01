use std::{path::Path, ffi::OsStr, process::Command};
use anyhow::{Result, anyhow};
use codegen::Compiler;
use continuous::ModelInfo;
use discretise::DiscreteModel;
use parser::{parse_ms_string, parse_ds_string};
use utils::{find_executable, find_runtime_path};

extern crate pest;
#[macro_use]
extern crate pest_derive;

pub mod parser;
pub mod ast;
pub mod discretise;
pub mod continuous;
pub mod codegen;
pub mod utils;


pub struct CompilerOptions {
    pub bitcode_only: bool,
    pub wasm: bool,
    pub standalone: bool,
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
    let out = out.unwrap_or("out");
    let text = std::fs::read_to_string(inputfile)?;
    compile_text(text.as_str(), out, model_name, options, is_discrete)
}

pub fn compile_text(text: &str, out: &str, model_name: &str, options: CompilerOptions, is_discrete: bool) -> Result<()> {
    let CompilerOptions { bitcode_only, wasm, standalone } = options;
    
    let is_continuous = !is_discrete;
    
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
    let compiler = Compiler::from_discrete_model(&discrete_model, out)?;

    // if we are only compiling to bitcode , we are done
    if bitcode_only {
        return Ok(());
    }

    let bitcode_filename = compiler.get_bitcode_filename();
    
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
        command.arg("-o").arg(out).arg(bitcode_filename);
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

#[cfg(test)]
mod tests {
    use crate::{parser::{parse_ds_string, parse_ms_string}, continuous::ModelInfo};
    use approx::assert_relative_eq;

    use super::*;

    fn ds_example_compiler(example: &str) -> Compiler {
        let text = std::fs::read_to_string(format!("examples/{}.ds", example)).unwrap();
        let model = parse_ds_string(text.as_str()).unwrap();
        let model = DiscreteModel::build(example, &model).unwrap_or_else(|e| panic!("{}", e.as_error_message(text.as_str())));
        let out = format!("lib_examples_{}", example);
        Compiler::from_discrete_model(&model, out.as_str()).unwrap()
    }

    #[test]
    fn test_logistic_ds_example() {
        let compiler = ds_example_compiler("logistic");
        let r = 0.5;
        let k = 0.5;
        let y = 0.5;
        let dydt = r * y * (1. - y / k);
        let z = 2. * y;
        let dzdt = 2. * dydt;
        let inputs = vec![r, k];
        let mut u0 = vec![y, z];
        let mut up0 = vec![dydt, dzdt];
        let mut data = compiler.get_new_data();
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice()).unwrap();
        compiler.set_u0(u0.as_mut_slice(), up0.as_mut_slice(), data.as_mut_slice()).unwrap();

        u0 = vec![y, z];
        up0 = vec![dydt, dzdt];
        let mut res = vec![1., 1.];

        compiler.residual(0., u0.as_slice(), up0.as_slice(), data.as_mut_slice(), res.as_mut_slice()).unwrap();
        let expected_value = vec![0., 0.];
        assert_relative_eq!(res.as_slice(), expected_value.as_slice());
    }

    #[test]
    fn test_object_file() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), z(t)) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
            z = 2 * y
        }
        ";
        let models = parse_ms_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete_model = DiscreteModel::from(&model_info);
        let object = Compiler::from_discrete_model(&discrete_model, "lib_test_object_file").unwrap();
        let path = Path::new("main.o");
        object.write_object_file(path).unwrap();
    }
}





