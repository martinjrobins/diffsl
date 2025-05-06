use js_sys::{Function, Map, Object, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use diffsl::execution::functions::*;


fn bind(this: &JsValue, func_name: &str) -> Result<(), JsValue> {
    let property_key = JsValue::from(func_name);
    let orig_func = Reflect::get(this, &property_key)?.dyn_into::<Function>()?;
    let func = orig_func.bind(this);
    if !Reflect::set(this, &property_key, &func)? {
        return Err(JsValue::from("failed to set property"));
    }
    Ok(())
}

pub fn make_imports() -> Result<Object, JsValue> {
    let map = Map::new();
    let imports: JsValue = Imports.into();

    // add supported external rust functions
    for func in FUNCTIONS.iter() {
        bind(&imports, func.0)?;
        bind(&imports, format!("d{}", func.0).as_str())?;
    }
    for func in TWO_ARG_FUNCTIONS.iter() {
        bind(&imports, func.0)?;
        bind(&imports, format!("d{}", func.0).as_str())?;
    }

    map.set(&JsValue::from("env"), &imports);
    Object::from_entries(&map.into())
}

#[wasm_bindgen]
pub struct Imports;

#[wasm_bindgen]
impl Imports {
    pub fn sin(&self, x: f64) -> f64 {
        sin(x)
    }
    pub fn dsin(&self, x: f64, dx: f64) -> f64 {
        dsin(x, dx)
    }
    pub fn cos(&self, x: f64) -> f64 {
        cos(x)
    }
    pub fn dcos(&self, x: f64, dx: f64) -> f64 {
        dcos(x, dx)
    }
    pub fn tan(&self, x: f64) -> f64 {
        tan(x)
    }
    pub fn dtan(&self, x: f64, dx: f64) -> f64 {
        dtan(x, dx)
    }
    pub fn exp(&self, x: f64) -> f64 {
        exp(x)
    }
    pub fn dexp(&self, x: f64, dx: f64) -> f64 {
        dexp(x, dx)
    }
    pub fn log(&self, x: f64) -> f64 {
        log(x)
    }
    pub fn dlog(&self, x: f64, dx: f64) -> f64 {
        dlog(x, dx)
    }
    pub fn log10(&self, x: f64) -> f64 {
        log10(x)
    }
    pub fn dlog10(&self, x: f64, dx: f64) -> f64 {
        dlog10(x, dx)
    }
    pub fn sqrt(&self, x: f64) -> f64 {
        sqrt(x)
    }
    pub fn dsqrt(&self, x: f64, dx: f64) -> f64 {
        dsqrt(x, dx)
    }
    pub fn abs(&self, x: f64) -> f64 {
        abs(x)
    }
    pub fn dabs(&self, x: f64, dx: f64) -> f64 {
        dabs(x, dx)
    }
    pub fn sigmoid(&self, x: f64) -> f64 {
        sigmoid(x)
    }
    pub fn dsigmoid(&self, x: f64, dx: f64) -> f64 {
        dsigmoid(x, dx)
    }
    pub fn arcsinh(&self, x: f64) -> f64 {
        arcsinh(x)
    }
    pub fn darcsinh(&self, x: f64, dx: f64) -> f64 {
        darcsinh(x, dx)
    }
    pub fn arccosh(&self, x: f64) -> f64 {
        arccosh(x)
    }
    pub fn darccosh(&self, x: f64, dx: f64) -> f64 {
        darccosh(x, dx)
    }
    pub fn heaviside(&self, x: f64) -> f64 {
        heaviside(x)
    }
    pub fn dheaviside(&self, x: f64, dx: f64) -> f64 {
        dheaviside(x, dx)
    }
    pub fn tanh(&self, x: f64) -> f64 {
        tanh(x)
    }
    pub fn dtanh(&self, x: f64, dx: f64) -> f64 {
        dtanh(x, dx)
    }
    pub fn sinh(&self, x: f64) -> f64 {
        sinh(x)
    }
    pub fn dsinh(&self, x: f64, dx: f64) -> f64 {
        dsinh(x, dx)
    }
    pub fn cosh(&self, x: f64) -> f64 {
        cosh(x)
    }
    pub fn dcosh(&self, x: f64, dx: f64) -> f64 {
        dcosh(x, dx)
    }
    pub fn copysign(&self, x: f64, y: f64) -> f64 {
        copysign(x, y)
    }
    pub fn dcopysign(&self, x: f64, dx: f64, y: f64, dy: f64) -> f64 {
        dcopysign(x, y, dx, dy)
    }
    pub fn pow(&self, x: f64, y: f64) -> f64 {
        pow(x, y)
    }
    pub fn dpow(&self, x: f64, dx: f64, y: f64, dy: f64) -> f64 {
        dpow(x, y, dx, dy)
    }
    pub fn min(&self, x: f64, y: f64) -> f64 {
        min(x, y)
    }
    pub fn dmin(&self, x: f64, dx: f64, y: f64, dy: f64) -> f64 {
        dmin(x, y, dx, dy)
    }
    pub fn max(&self, x: f64, y: f64) -> f64 {
        max(x, y)
    }
    pub fn dmax(&self, x: f64, dx: f64, y: f64, dy: f64) -> f64 {
        dmax(x, y, dx, dy)
    }
}
