use std::collections::HashMap;

use anyhow::Result;
use target_lexicon::Triple;

use crate::discretise::DiscreteModel;

use super::compiler::CompilerMode;

pub trait CodegenModule: Sized + Send + Sync + 'static {}

pub trait CodegenModuleCompile: CodegenModule {
    fn from_discrete_model(
        model: &DiscreteModel,
        mode: CompilerMode,
        triple: Option<Triple>,
    ) -> Result<Self>;
}

pub trait CodegenModuleLink: CodegenModule {
    fn from_object(buffer: &[u8]) -> Result<Self>;
}

pub trait CodegenModuleJit: CodegenModule {
    fn jit(&mut self) -> Result<HashMap<String, *const u8>>;
}

pub trait CodegenModuleEmit: CodegenModule {
    fn to_object(self) -> Result<Vec<u8>>;
}
