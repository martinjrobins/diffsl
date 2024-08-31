type RealType = f64;

pub type StopFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    root: *mut RealType,
);
pub type RhsFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    rr: *mut RealType,
);
pub type RhsGradientFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *mut RealType,
    ddata: *mut RealType,
    rr: *mut RealType,
    drr: *mut RealType,
);
pub type MassFunc = unsafe extern "C" fn(
    time: RealType,
    v: *const RealType,
    data: *mut RealType,
    mv: *mut RealType,
);
pub type U0Func = unsafe extern "C" fn(data: *mut RealType, u: *mut RealType);
pub type U0GradientFunc = unsafe extern "C" fn(
    data: *mut RealType,
    ddata: *mut RealType,
    u: *mut RealType,
    du: *mut RealType,
);
pub type CalcOutFunc =
    unsafe extern "C" fn(time: RealType, u: *const RealType, data: *mut RealType);
pub type CalcOutGradientFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *mut RealType,
    ddata: *mut RealType,
);
pub type GetDimsFunc = unsafe extern "C" fn(
    states: *mut u32,
    inputs: *mut u32,
    outputs: *mut u32,
    data: *mut u32,
    stop: *mut u32,
);
pub type SetInputsFunc = unsafe extern "C" fn(inputs: *const RealType, data: *mut RealType);
pub type SetInputsGradientFunc = unsafe extern "C" fn(
    inputs: *const RealType,
    dinputs: *const RealType,
    data: *mut RealType,
    ddata: *mut RealType,
);
pub type SetIdFunc = unsafe extern "C" fn(id: *mut RealType);
pub type GetOutFunc = unsafe extern "C" fn(
    data: *const RealType,
    tensor_data: *mut *mut RealType,
    tensor_size: *mut u32,
);
