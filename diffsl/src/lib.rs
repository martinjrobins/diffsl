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
pub use execution::cranelift::codegen::{CraneliftJitModule, CraneliftObjectModule};
pub use execution::external::ExternalModule;
#[cfg(feature = "llvm")]
pub use execution::llvm::codegen::LlvmModule;
pub use execution::module::{
    CodegenModule, CodegenModuleCompile, CodegenModuleEmit, CodegenModuleJit, CodegenModuleLink,
};
pub use execution::object::ObjectModule;

#[cfg(feature = "inkwell-150")]
extern crate inkwell_150 as inkwell;
#[cfg(feature = "inkwell-160")]
extern crate inkwell_160 as inkwell;
#[cfg(feature = "inkwell-170")]
extern crate inkwell_170 as inkwell;
#[cfg(feature = "inkwell-181")]
extern crate inkwell_181 as inkwell;
#[cfg(feature = "inkwell-191")]
extern crate inkwell_191 as inkwell;
#[cfg(feature = "inkwell-201")]
extern crate inkwell_201 as inkwell;
#[cfg(feature = "inkwell-211")]
extern crate inkwell_211 as inkwell;

#[cfg(feature = "inkwell-150")]
extern crate llvm_sys_150 as llvm_sys;
#[cfg(feature = "inkwell-160")]
extern crate llvm_sys_160 as llvm_sys;
#[cfg(feature = "inkwell-170")]
extern crate llvm_sys_170 as llvm_sys;
#[cfg(feature = "inkwell-181")]
extern crate llvm_sys_181 as llvm_sys;
#[cfg(feature = "inkwell-191")]
extern crate llvm_sys_191 as llvm_sys;
#[cfg(feature = "inkwell-201")]
extern crate llvm_sys_201 as llvm_sys;
#[cfg(feature = "inkwell-211")]
extern crate llvm_sys_211 as llvm_sys;
