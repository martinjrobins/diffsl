#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use llvm_sys::prelude::{LLVMBuilderRef, LLVMValueRef};
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
