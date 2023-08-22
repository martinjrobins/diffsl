
use std::{path::Path, ffi::OsStr};

use clap::Parser;
use diffeq::{parser::{parse_ms_string, parse_ds_string}, continuous::ModelInfo, discretise::DiscreteModel, codegen::Compiler};

/// compiles a model in continuous (.cs) or discrete (.ds) format to an object file
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input filename
    input: String,

    /// Output filename
    #[arg(short, long)]
    out: Option<String>,
    
    /// Model to build (only for continuous model files)
    #[arg(short, long)]
    model: Option<String>,
}

fn main() {
    let cli = Args::parse();

    let inputfile = Path::new(&cli.input);
    let out = if let Some(out) = cli.out {
        out.clone()
    } else {
        "out.o".to_owned()
    };
    let outfile = Path::new(&out);
    let is_discrete = inputfile.extension().unwrap_or(OsStr::new("")).to_str().unwrap() == "ds";
    let is_continuous = inputfile.extension().unwrap_or(OsStr::new("")).to_str().unwrap() == "cs";
    if !is_discrete && !is_continuous {
        panic!("Input file must have extension .ds or .cs");
    }
    let text = match std::fs::read_to_string(inputfile) {
        Ok(text) => {
            text
        }
        Err(e) => {
            panic!("{}", e);
        }
    };
    if is_continuous {
        let models = match parse_ms_string(text.as_str()) {
            Ok(models) => {
                models
            }
            Err(e) => {
                panic!("{}", e);
            }
        };
        let model_name = if let Some(model_name) = cli.model {
            model_name
        } else {
            panic!("Model name must be specified for continuous models");
        };
        let model_info = match ModelInfo::build(model_name.as_str(), &models) {
            Ok(model_info) => {
                model_info
            }
            Err(e) => {
                panic!("{}", e);
            }
        };
        if model_info.errors.len() > 0 {
            for error in model_info.errors {
                println!("{}", error.as_error_message(text.as_str()));
            }
            panic!("Errors in model");
        }
        let model = DiscreteModel::from(&model_info);
        let compiler = match Compiler::from_discrete_model(&model) {
            Ok(compiler) => {
                compiler
            }
            Err(e) => {
                panic!("{}", e);
            }
        };
        compiler.write_object_file(outfile).unwrap();
    } else {
        let model = match parse_ds_string(text.as_str()) {
            Ok(model) => {
                model
            }
            Err(e) => {
                panic!("{}", e);
            }
        };
        let model = match DiscreteModel::build(&cli.input, &model) {
            Ok(model) => {
                model
            }
            Err(e) => {
                panic!("{}", e.as_error_message(text.as_str()));
            }
        };
        let compiler = match Compiler::from_discrete_model(&model) {
            Ok(compiler) => {
                compiler
            }
            Err(e) => {
                panic!("{}", e);
            }
        };
        compiler.write_object_file(outfile).unwrap();
    };
}
     