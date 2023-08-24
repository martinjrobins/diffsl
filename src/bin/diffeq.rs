
use std::{path::Path, ffi::OsStr, process::Command};

use clap::Parser;
use anyhow::{Result, anyhow};
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
    
    /// Compile object file only
    #[arg(short, long)]
    compile: bool,
}

fn main() -> Result<()> {
    let cli = Args::parse();

    let inputfile = Path::new(&cli.input);
    let out = if let Some(out) = cli.out {
        out.clone()
    } else if cli.compile {
        "out.o".to_owned()
    } else {
        "out".to_owned()
    };

    let objectname = if cli.compile { out.clone() } else { format!("{}.o", out) };
    let objectfile = Path::new(objectname.as_str());
    let is_discrete = inputfile.extension().unwrap_or(OsStr::new("")).to_str().unwrap() == "ds";
    let is_continuous = inputfile.extension().unwrap_or(OsStr::new("")).to_str().unwrap() == "cs";
    if !is_discrete && !is_continuous {
        panic!("Input file must have extension .ds or .cs");
    }
    let text = std::fs::read_to_string(inputfile)?;
    if is_continuous {
        let models = parse_ms_string(text.as_str())?;
        let model_name = if let Some(model_name) = cli.model {
            model_name
        } else {
            return Err(anyhow!("Model name must be specified for continuous models"));
        };
        let model_info = ModelInfo::build(model_name.as_str(), &models).map_err(|e| anyhow!("{}", e))?;
        if model_info.errors.len() > 0 {
            for error in model_info.errors {
                println!("{}", error.as_error_message(text.as_str()));
            }
            return Err(anyhow!("Errors in model"));
        }
        let model = DiscreteModel::from(&model_info);
        let compiler = Compiler::from_discrete_model(&model)?;
        compiler.write_object_file(objectfile)?;
    } else {
        let model = parse_ds_string(text.as_str())?;
        let model = match DiscreteModel::build(&cli.input, &model) {
            Ok(model) => model,
            Err(e) => {
                println!("{}", e.as_error_message(text.as_str()));
                return Err(anyhow!("Errors in model"));
            }
        };
        let compiler = Compiler::from_discrete_model(&model)?;
        compiler.write_object_file(objectfile)?;
    };
    
    if cli.compile {
        return Ok(());
    }
    
    // compile the object file using clang and our runtime library
    let output = Command::new("clang")
            .arg("-o")
            .arg(out)
            .arg(objectname.clone())
            .arg("-ldiffeq_runtime")
            .output()?;
    
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
     