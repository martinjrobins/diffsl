use anyhow::{anyhow, Result};
use std::collections::HashMap;

type UIntType = u32;

pub type BarrierInitFunc = unsafe extern "C" fn();

pub type SetConstantsFunc = unsafe extern "C" fn(thread_id: UIntType, thread_dim: UIntType);

pub type StopFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *mut T,
    root: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type StopGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *const T,
    data: *const T,
    ddata: *mut T,
    root: *const T,
    droot: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type StopRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *mut T,
    data: *const T,
    ddata: *mut T,
    root: *const T,
    droot: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type StopSensGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    root: *const T,
    droot: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type StopSensRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    root: *const T,
    droot: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type ResetFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *mut T,
    reset: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type ResetGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *const T,
    data: *const T,
    ddata: *mut T,
    reset: *const T,
    dreset: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type ResetRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *mut T,
    data: *const T,
    ddata: *mut T,
    reset: *const T,
    dreset: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type ResetSensGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    reset: *const T,
    dreset: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type ResetSensRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    reset: *const T,
    dreset: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *mut T,
    rr: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *const T,
    data: *const T,
    ddata: *mut T,
    rr: *const T,
    drr: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *mut T,
    data: *const T,
    ddata: *mut T,
    rr: *const T,
    drr: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsSensGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    rr: *const T,
    drr: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type RhsSensRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    rr: *const T,
    drr: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type MassFunc<T> = unsafe extern "C" fn(
    time: T,
    v: *const T,
    data: *mut T,
    mv: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type MassRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    v: *const T,
    dv: *mut T,
    data: *const T,
    ddata: *mut T,
    mv: *const T,
    dmv: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type U0Func<T> =
    unsafe extern "C" fn(u: *mut T, data: *mut T, thread_id: UIntType, thread_dim: UIntType);
pub type U0SensGradFunc<T> = unsafe extern "C" fn(
    u: *const T,
    du: *mut T,
    data: *const T,
    ddata: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type U0GradFunc<T> = unsafe extern "C" fn(
    u: *const T,
    du: *mut T,
    data: *const T,
    ddata: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type U0RevGradFunc<T> = unsafe extern "C" fn(
    u: *const T,
    du: *mut T,
    data: *const T,
    ddata: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *mut T,
    out: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *const T,
    data: *const T,
    ddata: *mut T,
    out: *const T,
    dout: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    du: *mut T,
    data: *const T,
    ddata: *mut T,
    out: *const T,
    dout: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutSensGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    out: *const T,
    dout: *mut T,
    thread_id: UIntType,
    thread_dim: UIntType,
);
pub type CalcOutSensRevGradFunc<T> = unsafe extern "C" fn(
    time: T,
    u: *const T,
    data: *const T,
    ddata: *mut T,
    out: *const T,
    dout: *mut T,
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
    has_reset: *mut UIntType,
);
pub type SetInputsFunc<T> =
    unsafe extern "C" fn(inputs: *const T, data: *mut T, model_index: UIntType);
pub type GetInputsFunc<T> = unsafe extern "C" fn(inputs: *mut T, data: *const T);
pub type SetInputsGradFunc<T> = unsafe extern "C" fn(
    inputs: *const T,
    dinputs: *const T,
    data: *const T,
    ddata: *mut T,
    model_index: UIntType,
);
pub type SetInputsRevGradFunc<T> = unsafe extern "C" fn(
    inputs: *const T,
    dinputs: *mut T,
    data: *const T,
    ddata: *mut T,
    model_index: UIntType,
);
pub type SetIdFunc<T> = unsafe extern "C" fn(id: *mut T);
pub type GetTensorFunc<T> =
    unsafe extern "C" fn(data: *const T, tensor_data: *mut *mut T, tensor_size: *mut UIntType);
pub type GetConstantFunc<T> =
    unsafe extern "C" fn(tensor_data: *mut *const T, tensor_size: *mut UIntType);

pub(crate) struct JitFunctions<T> {
    pub(crate) set_u0: U0Func<T>,
    pub(crate) reset: ResetFunc<T>,
    pub(crate) rhs: RhsFunc<T>,
    pub(crate) mass: MassFunc<T>,
    pub(crate) calc_out: CalcOutFunc<T>,
    pub(crate) calc_stop: StopFunc<T>,
    pub(crate) set_id: SetIdFunc<T>,
    pub(crate) get_dims: GetDimsFunc,
    pub(crate) set_inputs: SetInputsFunc<T>,
    pub(crate) get_inputs: GetInputsFunc<T>,
    #[allow(dead_code)]
    pub(crate) barrier_init: Option<BarrierInitFunc>,
    pub(crate) set_constants: SetConstantsFunc,
}

impl<T> JitFunctions<T> {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        // check if all required symbols are present
        let required_symbols = [
            "set_u0",
            "reset",
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
        let set_u0 = unsafe { std::mem::transmute::<*const u8, U0Func<T>>(symbol_map["set_u0"]) };
        let reset = unsafe { std::mem::transmute::<*const u8, ResetFunc<T>>(symbol_map["reset"]) };
        let rhs = unsafe { std::mem::transmute::<*const u8, RhsFunc<T>>(symbol_map["rhs"]) };
        let mass = unsafe { std::mem::transmute::<*const u8, MassFunc<T>>(symbol_map["mass"]) };
        let calc_out =
            unsafe { std::mem::transmute::<*const u8, CalcOutFunc<T>>(symbol_map["calc_out"]) };
        let calc_stop =
            unsafe { std::mem::transmute::<*const u8, StopFunc<T>>(symbol_map["calc_stop"]) };
        let set_id =
            unsafe { std::mem::transmute::<*const u8, SetIdFunc<T>>(symbol_map["set_id"]) };
        let get_dims =
            unsafe { std::mem::transmute::<*const u8, GetDimsFunc>(symbol_map["get_dims"]) };
        let set_inputs =
            unsafe { std::mem::transmute::<*const u8, SetInputsFunc<T>>(symbol_map["set_inputs"]) };
        let get_inputs =
            unsafe { std::mem::transmute::<*const u8, GetInputsFunc<T>>(symbol_map["get_inputs"]) };
        let barrier_init = symbol_map.get("barrier_init").map(|func_ptr| unsafe {
            std::mem::transmute::<*const u8, BarrierInitFunc>(*func_ptr)
        });
        let set_constants = unsafe {
            std::mem::transmute::<*const u8, SetConstantsFunc>(symbol_map["set_constants"])
        };

        Ok(Self {
            set_u0,
            reset,
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

pub(crate) struct JitGradFunctions<T> {
    pub(crate) set_u0_grad: U0GradFunc<T>,
    pub(crate) stop_grad: StopGradFunc<T>,
    pub(crate) reset_grad: ResetGradFunc<T>,
    pub(crate) rhs_grad: RhsGradFunc<T>,
    pub(crate) calc_out_grad: CalcOutGradFunc<T>,
    pub(crate) set_inputs_grad: SetInputsGradFunc<T>,
}

impl<T> JitGradFunctions<T> {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        // check if all required symbols are present
        let required_symbols = [
            "set_u0_grad",
            "calc_stop_grad",
            "reset_grad",
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
            unsafe { std::mem::transmute::<*const u8, U0GradFunc<T>>(symbol_map["set_u0_grad"]) };
        let stop_grad = unsafe {
            std::mem::transmute::<*const u8, StopGradFunc<T>>(symbol_map["calc_stop_grad"])
        };
        let reset_grad =
            unsafe { std::mem::transmute::<*const u8, ResetGradFunc<T>>(symbol_map["reset_grad"]) };
        let rhs_grad =
            unsafe { std::mem::transmute::<*const u8, RhsGradFunc<T>>(symbol_map["rhs_grad"]) };
        let calc_out_grad = unsafe {
            std::mem::transmute::<*const u8, CalcOutGradFunc<T>>(symbol_map["calc_out_grad"])
        };
        let set_inputs_grad = unsafe {
            std::mem::transmute::<*const u8, SetInputsGradFunc<T>>(symbol_map["set_inputs_grad"])
        };

        Ok(Self {
            set_u0_grad,
            stop_grad,
            reset_grad,
            rhs_grad,
            calc_out_grad,
            set_inputs_grad,
        })
    }
}

pub(crate) struct JitGradRFunctions<T> {
    pub(crate) set_u0_rgrad: U0RevGradFunc<T>,
    pub(crate) stop_rgrad: StopRevGradFunc<T>,
    pub(crate) reset_rgrad: ResetRevGradFunc<T>,
    pub(crate) rhs_rgrad: RhsRevGradFunc<T>,
    pub(crate) mass_rgrad: MassRevGradFunc<T>,
    pub(crate) calc_out_rgrad: CalcOutRevGradFunc<T>,
    pub(crate) set_inputs_rgrad: SetInputsRevGradFunc<T>,
}

impl<T> JitGradRFunctions<T> {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let required_symbols = [
            "set_u0_rgrad",
            "calc_stop_rgrad",
            "reset_rgrad",
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
        let set_u0_rgrad = unsafe {
            std::mem::transmute::<*const u8, U0RevGradFunc<T>>(symbol_map["set_u0_rgrad"])
        };
        let stop_rgrad = unsafe {
            std::mem::transmute::<*const u8, StopRevGradFunc<T>>(symbol_map["calc_stop_rgrad"])
        };
        let reset_rgrad = unsafe {
            std::mem::transmute::<*const u8, ResetRevGradFunc<T>>(symbol_map["reset_rgrad"])
        };
        let rhs_rgrad =
            unsafe { std::mem::transmute::<*const u8, RhsRevGradFunc<T>>(symbol_map["rhs_rgrad"]) };
        let mass_rgrad = unsafe {
            std::mem::transmute::<*const u8, MassRevGradFunc<T>>(symbol_map["mass_rgrad"])
        };
        let calc_out_rgrad = unsafe {
            std::mem::transmute::<*const u8, CalcOutRevGradFunc<T>>(symbol_map["calc_out_rgrad"])
        };
        let set_inputs_rgrad = unsafe {
            std::mem::transmute::<*const u8, SetInputsRevGradFunc<T>>(
                symbol_map["set_inputs_rgrad"],
            )
        };

        Ok(Self {
            set_u0_rgrad,
            stop_rgrad,
            reset_rgrad,
            rhs_rgrad,
            mass_rgrad,
            calc_out_rgrad,
            set_inputs_rgrad,
        })
    }
}

pub(crate) struct JitSensGradFunctions<T> {
    pub(crate) set_u0_sgrad: U0SensGradFunc<T>,
    pub(crate) stop_sgrad: StopSensGradFunc<T>,
    pub(crate) reset_sgrad: ResetSensGradFunc<T>,
    pub(crate) rhs_sgrad: RhsSensGradFunc<T>,
    pub(crate) calc_out_sgrad: CalcOutSensGradFunc<T>,
}

impl<T> JitSensGradFunctions<T> {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let required_symbols = [
            "rhs_sgrad",
            "calc_out_sgrad",
            "set_u0_sgrad",
            "calc_stop_sgrad",
            "reset_sgrad",
        ];
        for symbol in &required_symbols {
            if !symbol_map.contains_key(*symbol) {
                return Err(anyhow!("Missing required symbol: {}", symbol));
            }
        }
        let rhs_sgrad = unsafe {
            std::mem::transmute::<*const u8, RhsSensGradFunc<T>>(symbol_map["rhs_sgrad"])
        };
        let calc_out_sgrad = unsafe {
            std::mem::transmute::<*const u8, CalcOutSensGradFunc<T>>(symbol_map["calc_out_sgrad"])
        };
        let set_u0_sgrad = unsafe {
            std::mem::transmute::<*const u8, U0SensGradFunc<T>>(symbol_map["set_u0_sgrad"])
        };
        let stop_sgrad = unsafe {
            std::mem::transmute::<*const u8, StopSensGradFunc<T>>(symbol_map["calc_stop_sgrad"])
        };
        let reset_sgrad = unsafe {
            std::mem::transmute::<*const u8, ResetSensGradFunc<T>>(symbol_map["reset_sgrad"])
        };

        Ok(Self {
            rhs_sgrad,
            calc_out_sgrad,
            set_u0_sgrad,
            stop_sgrad,
            reset_sgrad,
        })
    }
}

pub(crate) struct JitSensRevGradFunctions<T> {
    pub(crate) stop_rgrad: StopSensRevGradFunc<T>,
    pub(crate) reset_rgrad: ResetSensRevGradFunc<T>,
    pub(crate) rhs_rgrad: RhsSensRevGradFunc<T>,
    pub(crate) calc_out_rgrad: CalcOutSensRevGradFunc<T>,
}

impl<T> JitSensRevGradFunctions<T> {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let required_symbols = [
            "rhs_srgrad",
            "calc_out_srgrad",
            "calc_stop_srgrad",
            "reset_srgrad",
        ];
        for symbol in &required_symbols {
            if !symbol_map.contains_key(*symbol) {
                return Err(anyhow!("Missing required symbol: {}", symbol));
            }
        }
        let reset_rgrad = unsafe {
            std::mem::transmute::<*const u8, ResetSensRevGradFunc<T>>(symbol_map["reset_srgrad"])
        };
        let stop_rgrad = unsafe {
            std::mem::transmute::<*const u8, StopSensRevGradFunc<T>>(symbol_map["calc_stop_srgrad"])
        };
        let rhs_rgrad = unsafe {
            std::mem::transmute::<*const u8, RhsSensRevGradFunc<T>>(symbol_map["rhs_srgrad"])
        };
        let calc_out_rgrad = unsafe {
            std::mem::transmute::<*const u8, CalcOutSensRevGradFunc<T>>(
                symbol_map["calc_out_srgrad"],
            )
        };

        Ok(Self {
            stop_rgrad,
            reset_rgrad,
            rhs_rgrad,
            calc_out_rgrad,
        })
    }
}

pub(crate) struct JitGetTensorFunctions<T> {
    pub(crate) data_map: HashMap<String, GetTensorFunc<T>>,
    pub(crate) constant_map: HashMap<String, GetConstantFunc<T>>,
}

impl<T> JitGetTensorFunctions<T> {
    pub(crate) fn new(symbol_map: &HashMap<String, *const u8>) -> Result<Self> {
        let mut data_map = HashMap::new();
        let mut constant_map = HashMap::new();
        let data_prefix = "get_tensor_";
        let constant_prefix = "get_constant_";
        for (name, func_ptr) in symbol_map.iter() {
            if name.starts_with(data_prefix) {
                let func = unsafe { std::mem::transmute::<*const u8, GetTensorFunc<T>>(*func_ptr) };
                data_map.insert(name.strip_prefix(data_prefix).unwrap().to_string(), func);
            } else if name.starts_with(constant_prefix) {
                let func =
                    unsafe { std::mem::transmute::<*const u8, GetConstantFunc<T>>(*func_ptr) };
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
