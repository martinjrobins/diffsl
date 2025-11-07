use std::{collections::HashMap};

use js_sys::{Object, Reflect, WebAssembly::{self, Instance, Table}};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use anyhow::Result;

use crate::{CodegenModule, CodegenModuleJit, execution::interface::{RealType, UIntType}};


pub struct WasmJitModule;

impl WasmJitModule {
    pub async fn new(code: &[u8]) -> Self {
        println!("Compiling wasm module...");
        let env = Object::new();
        Reflect::set(&env, &"__linear_memory".into(), &wasm_bindgen::memory()).unwrap();
        Reflect::set(&env, &"__indirect_function_table".into(), &wasm_bindgen::function_table()).unwrap();
        let imports = Object::new();
        Reflect::set(&imports, &"env".into(), &env).unwrap();
        let compiled_wasm = JsFuture::from(WebAssembly::instantiate_buffer(code, &imports))
            .await
            .unwrap();
        let _instance = Reflect::get(&compiled_wasm, &"instance".into())
            .unwrap()
            .dyn_into::<Instance>()
            .unwrap();
        println!(" done.");
        WasmJitModule 
    }
}

impl CodegenModule for WasmJitModule {}

impl CodegenModuleJit for WasmJitModule {
    fn jit(&mut self) -> Result<HashMap<String, *const u8>> {
        let mut symbols = HashMap::new();
        let ftable = wasm_bindgen::function_table().dyn_into::<Table>().unwrap();
        let ftable_len = Reflect::get(&ftable, &"length".into()).unwrap().as_f64().unwrap() as u32;
        Ok(symbols)
    }
}

