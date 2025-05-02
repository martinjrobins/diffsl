extern crate pest;
#[macro_use]
extern crate pest_derive;

pub mod ast;
pub mod continuous;
pub mod discretise;
#[cfg(feature = "enzyme")]
pub mod enzyme;
pub mod execution;
pub mod parser;
pub mod utils;

pub use execution::compiler::Compiler;
#[cfg(feature = "cranelift")]
pub use execution::cranelift::codegen::CraneliftModule;
#[cfg(feature = "llvm")]
pub use execution::llvm::codegen::LlvmModule;

#[cfg(feature = "inkwell-150")]
extern crate inkwell_150 as inkwell;
#[cfg(feature = "inkwell-160")]
extern crate inkwell_160 as inkwell;
#[cfg(feature = "inkwell-170")]
extern crate inkwell_170 as inkwell;
#[cfg(feature = "inkwell-180")]
extern crate inkwell_180 as inkwell;

#[cfg(feature = "inkwell-150")]
extern crate llvm_sys_150 as llvm_sys;
#[cfg(feature = "inkwell-160")]
extern crate llvm_sys_160 as llvm_sys;
#[cfg(feature = "inkwell-170")]
extern crate llvm_sys_170 as llvm_sys;
#[cfg(feature = "inkwell-180")]
extern crate llvm_sys_180 as llvm_sys;
