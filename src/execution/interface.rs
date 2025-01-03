type RealType = f64;
type UIntType = u32;

pub type BarrierInitFunc = unsafe extern "C" fn();

pub type StopFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    root: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type RhsFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    rr: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type RhsGradientFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *mut RealType,
    ddata: *mut RealType,
    rr: *mut RealType,
    drr: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type MassFunc = unsafe extern "C" fn(
    time: RealType,
    v: *const RealType,
    data: *mut RealType,
    mv: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type U0Func = unsafe extern "C" fn(
    u: *mut RealType,
    data: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type U0GradientFunc = unsafe extern "C" fn(
    u: *mut RealType,
    du: *mut RealType,
    data: *mut RealType,
    ddata: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type CalcOutFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type CalcOutGradientFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *mut RealType,
    ddata: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
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
    tensor_size: *mut UIntType,
);
