use anyhow::{anyhow, Result};
use std::collections::HashMap;

type RealType = f64;
type UIntType = u32;

pub type BarrierInitFunc = unsafe extern "C" fn();

pub type SetConstantsFunc = unsafe extern "C" fn(thread_id: UIntType, thread_dim: UIntType);

pub type StopFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    root: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    rr: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsRevGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsSensGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsSensRevGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type MassFunc = unsafe extern "C" fn(
    time: RealType,
    v: *const RealType,
    data: *mut RealType,
    mv: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type MassRevGradFunc = unsafe extern "C" fn(
    time: RealType,
    v: *const RealType,
    dv: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
    mv: *const RealType,
    dmv: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type U0Func = unsafe extern "C" fn(
    u: *mut RealType,
    data: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type U0SensGradFunc = unsafe extern "C" fn(
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type U0GradFunc = unsafe extern "C" fn(
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type U0RevGradFunc = unsafe extern "C" fn(
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    out: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    out: *const RealType,
    dout: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutRevGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
    out: *const RealType,
    dout: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutSensGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    out: *const RealType,
    dout: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutSensRevGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    out: *const RealType,
    dout: *mut RealType,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type GetDimsFunc = unsafe extern "C" fn(
    states: *mut UIntType,
    inputs: *mut UIntType,
    outputs: *mut UIntType,
    data: *mut UIntType,
    stop: *mut UIntType,
    has_mass: *mut UIntType,
);
pub type SetInputsFunc = unsafe extern "C" fn(inputs: *const RealType, data: *mut RealType);
pub type GetInputsFunc = unsafe extern "C" fn(inputs: *mut RealType, data: *const RealType);
pub type SetInputsGradFunc = unsafe extern "C" fn(
    inputs: *const RealType,
    dinputs: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
);
pub type SetInputsRevGradFunc = unsafe extern "C" fn(
    inputs: *const RealType,
    dinputs: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
);
pub type SetIdFunc = unsafe extern "C" fn(id: *mut RealType);
pub type GetTensorFunc = unsafe extern "C" fn(
    data: *const RealType,
    tensor_data: *mut *mut RealType,
    tensor_size: *mut UIntType,
);
pub type GetConstantFunc =
    unsafe extern "C" fn(tensor_data: *mut *const RealType, tensor_size: *mut UIntType);

pub(crate) struct JitFunctions {
    pub(crate) set_u0: U0Func,
    pub(crate) rhs: RhsFunc,
    pub(crate) mass: MassFunc,
    pub(crate) calc_out: CalcOutFunc,
    pub(crate) calc_stop: StopFunc,
    pub(crate) set_id: SetIdFunc,
    pub(crate) get_dims: GetDimsFunc,
    pub(crate) set_inputs: SetInputsFunc,
    pub(crate) get_inputs: GetInputsFunc,
    #[allow(dead_code)]
    pub(crate) barrier_init: Option<BarrierInitFunc>,
    pub(crate) set_constants: SetConstantsFunc,
}

impl JitFunctions {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        // check if all required symbols are present
        let required_symbols = [
            "set_u0",
            "rhs",
            "mass",
            "calc_out",
            "calc_stop",
            "set_id",
            "get_dims",
            "set_inputs",
            "get_inputs",
            "set_constants",
        ];
        for symbol in &required_symbols {
            if !symbol_map.contains_key(*symbol) {
                return Err(anyhow!("Missing required symbol: {}", symbol));
            }
        }
        let set_u0 = unsafe { std::mem::transmute::<*const u8, U0Func>(symbol_map["set_u0"]) };
        let rhs = unsafe { std::mem::transmute::<*const u8, RhsFunc>(symbol_map["rhs"]) };
        let mass = unsafe { std::mem::transmute::<*const u8, MassFunc>(symbol_map["mass"]) };
        let calc_out =
            unsafe { std::mem::transmute::<*const u8, CalcOutFunc>(symbol_map["calc_out"]) };
        let calc_stop =
            unsafe { std::mem::transmute::<*const u8, StopFunc>(symbol_map["calc_stop"]) };
        let set_id = unsafe { std::mem::transmute::<*const u8, SetIdFunc>(symbol_map["set_id"]) };
        let get_dims =
            unsafe { std::mem::transmute::<*const u8, GetDimsFunc>(symbol_map["get_dims"]) };
        let set_inputs =
            unsafe { std::mem::transmute::<*const u8, SetInputsFunc>(symbol_map["set_inputs"]) };
        let get_inputs =
            unsafe { std::mem::transmute::<*const u8, GetInputsFunc>(symbol_map["get_inputs"]) };
        let barrier_init = symbol_map.get("barrier_init").map(|func_ptr| unsafe {
            std::mem::transmute::<*const u8, BarrierInitFunc>(*func_ptr)
        });
        let set_constants = unsafe {
            std::mem::transmute::<*const u8, SetConstantsFunc>(symbol_map["set_constants"])
        };

        Ok(Self {
            set_u0,
            rhs,
            mass,
            calc_out,
            calc_stop,
            set_id,
            get_dims,
            set_inputs,
            get_inputs,
            barrier_init,
            set_constants,
        })
    }
}

pub(crate) struct JitGradFunctions {
    pub(crate) set_u0_grad: U0GradFunc,
    pub(crate) rhs_grad: RhsGradFunc,
    pub(crate) calc_out_grad: CalcOutGradFunc,
    pub(crate) set_inputs_grad: SetInputsGradFunc,
}

impl JitGradFunctions {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        // check if all required symbols are present
        let required_symbols = [
            "set_u0_grad",
            "rhs_grad",
            "calc_out_grad",
            "set_inputs_grad",
        ];
        for symbol in &required_symbols {
            if !symbol_map.contains_key(*symbol) {
                return Err(anyhow!("Missing required symbol: {}", symbol));
            }
        }
        let set_u0_grad =
            unsafe { std::mem::transmute::<*const u8, U0GradFunc>(symbol_map["set_u0_grad"]) };
        let rhs_grad =
            unsafe { std::mem::transmute::<*const u8, RhsGradFunc>(symbol_map["rhs_grad"]) };
        let calc_out_grad = unsafe {
            std::mem::transmute::<*const u8, CalcOutGradFunc>(symbol_map["calc_out_grad"])
        };
        let set_inputs_grad = unsafe {
            std::mem::transmute::<*const u8, SetInputsGradFunc>(symbol_map["set_inputs_grad"])
        };

        Ok(Self {
            set_u0_grad,
            rhs_grad,
            calc_out_grad,
            set_inputs_grad,
        })
    }
}

pub(crate) struct JitGradRFunctions {
    pub(crate) set_u0_rgrad: U0RevGradFunc,
    pub(crate) rhs_rgrad: RhsRevGradFunc,
    pub(crate) mass_rgrad: MassRevGradFunc,
    pub(crate) calc_out_rgrad: CalcOutRevGradFunc,
    pub(crate) set_inputs_rgrad: SetInputsRevGradFunc,
}

impl JitGradRFunctions {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let required_symbols = [
            "set_u0_rgrad",
            "rhs_rgrad",
            "mass_rgrad",
            "calc_out_rgrad",
            "set_inputs_rgrad",
        ];
        for symbol in &required_symbols {
            if !symbol_map.contains_key(*symbol) {
                return Err(anyhow!("Missing required symbol: {}", symbol));
            }
        }
        let set_u0_rgrad =
            unsafe { std::mem::transmute::<*const u8, U0RevGradFunc>(symbol_map["set_u0_rgrad"]) };
        let rhs_rgrad =
            unsafe { std::mem::transmute::<*const u8, RhsRevGradFunc>(symbol_map["rhs_rgrad"]) };
        let mass_rgrad =
            unsafe { std::mem::transmute::<*const u8, MassRevGradFunc>(symbol_map["mass_rgrad"]) };
        let calc_out_rgrad = unsafe {
            std::mem::transmute::<*const u8, CalcOutRevGradFunc>(symbol_map["calc_out_rgrad"])
        };
        let set_inputs_rgrad = unsafe {
            std::mem::transmute::<*const u8, SetInputsRevGradFunc>(symbol_map["set_inputs_rgrad"])
        };

        Ok(Self {
            set_u0_rgrad,
            rhs_rgrad,
            mass_rgrad,
            calc_out_rgrad,
            set_inputs_rgrad,
        })
    }
}

pub(crate) struct JitSensGradFunctions {
    pub(crate) set_u0_sgrad: U0SensGradFunc,
    pub(crate) rhs_sgrad: RhsSensGradFunc,
    pub(crate) calc_out_sgrad: CalcOutSensGradFunc,
}

impl JitSensGradFunctions {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let required_symbols = ["rhs_sgrad", "calc_out_sgrad", "set_u0_sgrad"];
        for symbol in &required_symbols {
            if !symbol_map.contains_key(*symbol) {
                return Err(anyhow!("Missing required symbol: {}", symbol));
            }
        }
        let rhs_sgrad =
            unsafe { std::mem::transmute::<*const u8, RhsSensGradFunc>(symbol_map["rhs_sgrad"]) };
        let calc_out_sgrad = unsafe {
            std::mem::transmute::<*const u8, CalcOutSensGradFunc>(symbol_map["calc_out_sgrad"])
        };
        let set_u0_sgrad =
            unsafe { std::mem::transmute::<*const u8, U0SensGradFunc>(symbol_map["set_u0_sgrad"]) };

        Ok(Self {
            rhs_sgrad,
            calc_out_sgrad,
            set_u0_sgrad,
        })
    }
}

pub(crate) struct JitSensRevGradFunctions {
    pub(crate) rhs_rgrad: RhsSensRevGradFunc,
    pub(crate) calc_out_rgrad: CalcOutSensRevGradFunc,
}

impl JitSensRevGradFunctions {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let required_symbols = ["rhs_srgrad", "calc_out_srgrad"];
        for symbol in &required_symbols {
            if !symbol_map.contains_key(*symbol) {
                return Err(anyhow!("Missing required symbol: {}", symbol));
            }
        }
        let rhs_rgrad = unsafe {
            std::mem::transmute::<*const u8, RhsSensRevGradFunc>(symbol_map["rhs_srgrad"])
        };
        let calc_out_rgrad = unsafe {
            std::mem::transmute::<*const u8, CalcOutSensRevGradFunc>(symbol_map["calc_out_srgrad"])
        };

        Ok(Self {
            rhs_rgrad,
            calc_out_rgrad,
        })
    }
}

pub(crate) struct JitGetTensorFunctions {
    pub(crate) data_map: HashMap<String, GetTensorFunc>,
    pub(crate) constant_map: HashMap<String, GetConstantFunc>,
}

impl JitGetTensorFunctions {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let mut data_map = HashMap::new();
        let mut constant_map = HashMap::new();
        let data_prefix = "get_tensor_";
        let constant_prefix = "get_constant_";
        for (name, func_ptr) in symbol_map.iter() {
            if name.starts_with(data_prefix) {
                let func = unsafe { std::mem::transmute::<*const u8, GetTensorFunc>(*func_ptr) };
                data_map.insert(name.strip_prefix(data_prefix).unwrap().to_string(), func);
            } else if name.starts_with(constant_prefix) {
                let func = unsafe { std::mem::transmute::<*const u8, GetConstantFunc>(*func_ptr) };
                constant_map.insert(
                    name.strip_prefix(constant_prefix).unwrap().to_string(),
                    func,
                );
            }
        }
        Ok(Self {
            data_map,
            constant_map,
        })
    }
}
