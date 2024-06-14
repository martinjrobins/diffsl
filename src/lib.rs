use anyhow::{anyhow, Result};
use continuous::ModelInfo;
use discretise::DiscreteModel;
use execution::Compiler;
use parser::{parse_ds_string, parse_ms_string};
use std::{ffi::OsStr, path::Path};

extern crate pest;
#[macro_use]
extern crate pest_derive;

pub mod ast;
pub mod continuous;
pub mod discretise;
pub mod enzyme;
pub mod execution;
pub mod parser;
pub mod utils;

#[cfg(feature = "inkwell-100")]
pub extern crate inkwell_100 as inkwell;
#[cfg(feature = "inkwell-110")]
pub extern crate inkwell_110 as inkwell;
#[cfg(feature = "inkwell-120")]
pub extern crate inkwell_120 as inkwell;
#[cfg(feature = "inkwell-130")]
pub extern crate inkwell_130 as inkwell;
#[cfg(feature = "inkwell-140")]
pub extern crate inkwell_140 as inkwell;
#[cfg(feature = "inkwell-150")]
pub extern crate inkwell_150 as inkwell;
#[cfg(feature = "inkwell-160")]
pub extern crate inkwell_160 as inkwell;
#[cfg(feature = "inkwell-170")]
pub extern crate inkwell_170 as inkwell;
#[cfg(feature = "inkwell-40")]
pub extern crate inkwell_40 as inkwell;
#[cfg(feature = "inkwell-50")]
pub extern crate inkwell_50 as inkwell;
#[cfg(feature = "inkwell-60")]
pub extern crate inkwell_60 as inkwell;
#[cfg(feature = "inkwell-70")]
pub extern crate inkwell_70 as inkwell;
#[cfg(feature = "inkwell-80")]
pub extern crate inkwell_80 as inkwell;
#[cfg(feature = "inkwell-90")]
pub extern crate inkwell_90 as inkwell;
#[cfg(feature = "inkwell-140")]
pub extern crate llvm_sys_140 as llvm_sys;

pub struct CompilerOptions {
    pub bitcode_only: bool,
    pub wasm: bool,
    pub standalone: bool,
}

pub fn compile(
    input: &str,
    out: Option<&str>,
    model: Option<&str>,
    options: CompilerOptions,
) -> Result<()> {
    let inputfile = Path::new(input);
    let is_discrete = inputfile
        .extension()
        .unwrap_or(OsStr::new(""))
        .to_str()
        .unwrap()
        == "ds";
    let is_continuous = inputfile
        .extension()
        .unwrap_or(OsStr::new(""))
        .to_str()
        .unwrap()
        == "cs";
    if !is_discrete && !is_continuous {
        panic!("Input file must have extension .ds or .cs");
    }
    let model_name = if is_continuous {
        if let Some(model_name) = model {
            model_name
        } else {
            return Err(anyhow!(
                "Model name must be specified for continuous models"
            ));
        }
    } else {
        inputfile.file_stem().unwrap().to_str().unwrap()
    };
    let out = out.unwrap_or("out");
    let text = std::fs::read_to_string(inputfile)?;
    compile_text(text.as_str(), out, model_name, options, is_discrete)
}

pub fn compile_text(
    text: &str,
    out: &str,
    model_name: &str,
    options: CompilerOptions,
    is_discrete: bool,
) -> Result<()> {
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
        if !model_info.errors.is_empty() {
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
        let model = DiscreteModel::from(model_info);
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

    if options.bitcode_only {
        return Ok(());
    }

    compiler.compile(options.standalone, options.wasm)
}

#[cfg(test)]
mod tests {
    use crate::{
        continuous::ModelInfo,
        parser::{parse_ds_string, parse_ms_string},
    };
    use approx::assert_relative_eq;

    use super::*;

    fn ds_example_compiler(example: &str) -> Compiler {
        let text = std::fs::read_to_string(format!("examples/{}.ds", example)).unwrap();
        let model = parse_ds_string(text.as_str()).unwrap();
        let model = DiscreteModel::build(example, &model)
            .unwrap_or_else(|e| panic!("{}", e.as_error_message(text.as_str())));
        let out = format!("test_output/lib_examples_{}", example);
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
        let mut data = compiler.get_new_data();
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice());
        compiler.set_u0(u0.as_mut_slice(), data.as_mut_slice());

        u0 = vec![y, z];
        let up0 = vec![dydt, dzdt];
        let mut res = vec![1., 1.];

        compiler.rhs(0., u0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        let expected_value = vec![dydt, 2.0 * y - z];
        assert_relative_eq!(res.as_slice(), expected_value.as_slice());

        compiler.mass(0., up0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        let expected_value = vec![dydt, 0.];
        assert_relative_eq!(res.as_slice(), expected_value.as_slice());
    }

    #[test]
    fn test_heat2d_example() {
        let compiler = ds_example_compiler("heat2d");
        let inputs = vec![];
        let mut u0 = vec![0.0; 100];
        let mut data = compiler.get_new_data();
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice());
        compiler.set_u0(u0.as_mut_slice(), data.as_mut_slice());

        u0 = vec![0.0; 100];
        let up0 = vec![0.0; 100];
        let mut res = vec![0.0; 100];

        compiler.rhs(0., u0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        compiler.mass(0., up0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        //todo: check output or snapshot?
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
        let object =
            Compiler::from_discrete_model(&discrete_model, "test_output/lib_test_object_file")
                .unwrap();
        let path = Path::new("main.o");
        object.write_object_file(path).unwrap();
    }
}
