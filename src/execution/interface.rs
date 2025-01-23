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
pub type RhsGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type RhsRevGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *const RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type RhsSensGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *mut RealType,
    threadId: UIntType,
    threadDim: UIntType,
);
pub type RhsSensRevGradFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    data: *const RealType,
    ddata: *mut RealType,
    rr: *const RealType,
    drr: *const RealType,
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
pub type U0GradFunc = unsafe extern "C" fn(
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
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
pub type CalcOutReverseGradientFunc = unsafe extern "C" fn(
    time: RealType,
    u: *const RealType,
    du: *mut RealType,
    data: *const RealType,
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
