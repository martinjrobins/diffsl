use js_sys::{Function, Reflect, WebAssembly};
use wasm_bindgen::prelude::*;
use anyhow::Result;

#[wasm_bindgen]
pub struct Exports {
    instance: WebAssembly::Instance,

    stop: Function,
    rhs: Function,
    mass: Function,
    u0: Function,
    calc_out: Function,
    get_dims: Function,
    set_inputs: Function,
    get_inputs: Function,
    set_id: Function,
    get_tensor: Function,
    get_constant: Function,

    u0_grad: Function,
    rhs_grad: Function,
    calc_out_grad: Function,
    set_inputs_grad: Function,

    rhs_sens_grad: Option<Function>,
    calc_out_sens_grad: Option<Function>,

    mass_rev_grad: Option<Function>,
    rhs_rev_grad: Option<Function>,
    rhs_sens_rev_grad: Option<Function>,
    u0_rev_grad: Option<Function>,
    calc_out_rev_grad: Option<Function>,
    set_inputs_rev_grad: Option<Function>,
    calc_out_sens_rev_grad: Option<Function>,
}

#[wasm_bindgen]
impl Exports {
    pub fn new(instance: WebAssembly::Instance) -> Result<Self> {
        let exp = instance.exports();
        let stop = Reflect::get(exp.as_ref(), &"stop".into())?.dyn_into::<Function>().unwrap();
        let rhs = Reflect::get(exp.as_ref(), &"rhs".into())?.dyn_into::<Function>().unwrap();
        let mass = Reflect::get(exp.as_ref(), &"mass".into())?.dyn_into::<Function>().unwrap();
        let u0 = Reflect::get(exp.as_ref(), &"u0".into())?.dyn_into::<Function>().unwrap();
        let calc_out = Reflect::get(exp.as_ref(), &"calc_out".into())?.dyn_into::<Function>().unwrap();
        let get_dims = Reflect::get(exp.as_ref(), &"get_dims".into())?.dyn_into::<Function>().unwrap();
        let set_inputs = Reflect::get(exp.as_ref(), &"set_inputs".into())?.dyn_into::<Function>().unwrap();
        let get_inputs = Reflect::get(exp.as_ref(), &"get_inputs".into())?.dyn_into::<Function>().unwrap();
        let set_id = Reflect::get(exp.as_ref(), &"set_id".into())?.dyn_into::<Function>().unwrap();
        let get_tensor = Reflect::get(exp.as_ref(), &"get_tensor".into())?.dyn_into::<Function>().unwrap();
        let get_constant = Reflect::get(exp.as_ref(), &"get_constant".into())?.dyn_into::<Function>().unwrap();
        let u0_grad = Reflect::get(exp.as_ref(), &"u0_grad".into())?.dyn_into::<Function>().unwrap();
        let rhs_grad = Reflect::get(exp.as_ref(), &"rhs_grad".into())?.dyn_into::<Function>().unwrap();
        let calc_out_grad = Reflect::get(exp.as_ref(), &"calc_out_grad".into())?.dyn_into::<Function>().unwrap();
        let set_inputs_grad = Reflect::get(exp.as_ref(), &"set_inputs_grad".into())?.dyn_into::<Function>().unwrap();

        let rhs_sens_grad = Reflect::get(exp.as_ref(), &"rhs_sgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        let calc_out_sens_grad = Reflect::get(exp.as_ref(), &"calc_out_sgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();

        let mass_rev_grad = Reflect::get(exp.as_ref(), &"mass_rgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        let rhs_rev_grad = Reflect::get(exp.as_ref(), &"rhs_rgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        let rhs_sens_rev_grad = Reflect::get(exp.as_ref(), &"rhs_sens_rgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        let u0_rev_grad = Reflect::get(exp.as_ref(), &"u0_rgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        let calc_out_rev_grad = Reflect::get(exp.as_ref(), &"calc_out_rgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        let set_inputs_rev_grad = Reflect::get(exp.as_ref(), &"set_inputs_rgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        let calc_out_sens_rev_grad = Reflect::get(exp.as_ref(), &"calc_out_sens_rgrad".into()).map(|v| v.dyn_into::<Function>().unwrap()).ok();
        Ok(Exports {
            instance,
            stop,
            rhs,
            mass,
            u0,
            calc_out,
            get_dims,
            set_inputs,
            get_inputs,
            set_id,
            get_tensor,
            get_constant,
            u0_grad,
            rhs_grad,
            calc_out_grad,
            set_inputs_grad,
            rhs_sens_grad,
            calc_out_sens_grad,
            mass_rev_grad,
            rhs_rev_grad,
            rhs_sens_rev_grad,
            u0_rev_grad,
            calc_out_rev_grad,
            set_inputs_rev_grad,
            calc_out_sens_rev_grad
        })
    }

    pub fn stop(&self) -> Result<(), JsValue> {
        self.stop.call0(&JsValue::undefined())
    }
}