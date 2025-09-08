use js_sys::{Function, Map, Object, Reflect, WebAssembly};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use anyhow::Result;

use crate::imports::{self, make_imports};

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

async fn run_async(buffer: &[u8]) -> Result<(), JsValue> {
    console_log!("instantiating a new Wasm module directly");

    let imports = make_imports()?;
    let a = JsFuture::from(WebAssembly::instantiate_buffer(buffer, &imports)).await?;
    let b: WebAssembly::Instance = Reflect::get(&a, &"instance".into())?.dyn_into()?;

    let c = b.exports();

    let add = Reflect::get(c.as_ref(), &"add".into())?
        .dyn_into::<Function>()
        .expect("add export wasn't a function");

    let three = add.call2(&JsValue::undefined(), &1.into(), &2.into())?;
    console_log!("1 + 2 = {:?}", three);
    let mem = Reflect::get(c.as_ref(), &"memory".into())?
        .dyn_into::<WebAssembly::Memory>()
        .expect("memory export wasn't a `WebAssembly.Memory`");
    console_log!("created module has {} pages of memory", mem.grow(0));
    console_log!("giving the module 4 more pages of memory");
    mem.grow(4);
    console_log!("now the module has {} pages of memory", mem.grow(0));

    Ok(())
}