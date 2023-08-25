use clap::Parser;
use anyhow::Result;
use diffeq::compile;

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

    /// Compile to WASM
    #[arg(short, long)]
    wasm: bool,
}

fn main() -> Result<()> {
    let cli = Args::parse();
    compile(&cli.input, cli.out.as_deref(), cli.model.as_deref(), cli.compile, cli.wasm)
}
     