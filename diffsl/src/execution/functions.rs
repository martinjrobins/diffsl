#![allow(clippy::type_complexity)]
use crate::execution::scalar::RealType;

type UnaryFnF64 = extern "C" fn(f64) -> f64;
type UnaryFnF32 = extern "C" fn(f32) -> f32;
type UnaryGradFnF64 = extern "C" fn(f64, f64) -> f64;
type UnaryGradFnF32 = extern "C" fn(f32, f32) -> f32;
type BinaryFnF64 = extern "C" fn(f64, f64) -> f64;
type BinaryFnF32 = extern "C" fn(f32, f32) -> f32;
type BinaryGradFnF64 = extern "C" fn(f64, f64, f64, f64) -> f64;
type BinaryGradFnF32 = extern "C" fn(f32, f32, f32, f32) -> f32;

pub fn function_symbol_name(base_name: &str, real_type: RealType, is_tangent: bool) -> String {
    let suffix = real_type.as_str();
    let name = format!("{base_name}_{suffix}");
    if is_tangent {
        format!("{name}__tangent__")
    } else {
        name
    }
}

fn parse_function_name(raw: &str) -> (&str, bool, RealType) {
    let name = raw.strip_prefix('_').unwrap_or(raw);
    let (base_with_suffix, is_tangent) = match name.strip_suffix("__tangent__") {
        Some(base) => (base, true),
        None => (name, false),
    };

    if let Some(base) = base_with_suffix.strip_suffix("_f32") {
        return (base, is_tangent, RealType::F32);
    }
    if let Some(base) = base_with_suffix.strip_suffix("_f64") {
        return (base, is_tangent, RealType::F64);
    }

    // default to f64 for backward compatibility when there is no suffix
    (base_with_suffix, is_tangent, RealType::F64)
}

fn resolve_by_real_type(name: &str, is_tangent: bool, real_type: &RealType) -> Option<*const u8> {
    match real_type {
        RealType::F64 => FUNCTIONS_F64
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, f, df)| {
                if is_tangent {
                    *df as *const u8
                } else {
                    *f as *const u8
                }
            })
            .or_else(|| {
                TWO_ARG_FUNCTIONS_F64
                    .iter()
                    .find(|(n, _, _)| *n == name)
                    .map(|(_, f, df)| {
                        if is_tangent {
                            *df as *const u8
                        } else {
                            *f as *const u8
                        }
                    })
            }),
        RealType::F32 => FUNCTIONS_F32
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, f, df)| {
                if is_tangent {
                    *df as *const u8
                } else {
                    *f as *const u8
                }
            })
            .or_else(|| {
                TWO_ARG_FUNCTIONS_F32
                    .iter()
                    .find(|(n, _, _)| *n == name)
                    .map(|(_, f, df)| {
                        if is_tangent {
                            *df as *const u8
                        } else {
                            *f as *const u8
                        }
                    })
            }),
    }
}

pub const FUNCTIONS_F64: &[(&str, UnaryFnF64, UnaryGradFnF64)] = &[
    ("sin", sin_f64, dsin_f64),
    ("cos", cos_f64, dcos_f64),
    ("tan", tan_f64, dtan_f64),
    ("exp", exp_f64, dexp_f64),
    ("log", log_f64, dlog_f64),
    ("log10", log10_f64, dlog10_f64),
    ("sqrt", sqrt_f64, dsqrt_f64),
    ("abs", abs_f64, dabs_f64),
    ("sigmoid", sigmoid_f64, dsigmoid_f64),
    ("arcsinh", arcsinh_f64, darcsinh_f64),
    ("arccosh", arccosh_f64, darccosh_f64),
    ("heaviside", heaviside_f64, dheaviside_f64),
    ("tanh", tanh_f64, dtanh_f64),
    ("sinh", sinh_f64, dsinh_f64),
    ("cosh", cosh_f64, dcosh_f64),
];

pub const FUNCTIONS_F32: &[(&str, UnaryFnF32, UnaryGradFnF32)] = &[
    ("sin", sin_f32, dsin_f32),
    ("cos", cos_f32, dcos_f32),
    ("tan", tan_f32, dtan_f32),
    ("exp", exp_f32, dexp_f32),
    ("log", log_f32, dlog_f32),
    ("log10", log10_f32, dlog10_f32),
    ("sqrt", sqrt_f32, dsqrt_f32),
    ("abs", abs_f32, dabs_f32),
    ("sigmoid", sigmoid_f32, dsigmoid_f32),
    ("arcsinh", arcsinh_f32, darcsinh_f32),
    ("arccosh", arccosh_f32, darccosh_f32),
    ("heaviside", heaviside_f32, dheaviside_f32),
    ("tanh", tanh_f32, dtanh_f32),
    ("sinh", sinh_f32, dsinh_f32),
    ("cosh", cosh_f32, dcosh_f32),
];

// backward compatibility for existing callers expecting the old names
pub const FUNCTIONS: &[(&str, UnaryFnF64, UnaryGradFnF64)] = FUNCTIONS_F64;

pub const TWO_ARG_FUNCTIONS_F64: &[(&str, BinaryFnF64, BinaryGradFnF64)] = &[
    ("copysign", copysign_f64, dcopysign_f64),
    ("pow", pow_f64, dpow_f64),
    ("min", min_f64, dmin_f64),
    ("max", max_f64, dmax_f64),
];

pub const TWO_ARG_FUNCTIONS_F32: &[(&str, BinaryFnF32, BinaryGradFnF32)] = &[
    ("copysign", copysign_f32, dcopysign_f32),
    ("pow", pow_f32, dpow_f32),
    ("min", min_f32, dmin_f32),
    ("max", max_f32, dmax_f32),
];

// backward compatibility for existing callers expecting the old names
pub const TWO_ARG_FUNCTIONS: &[(&str, BinaryFnF64, BinaryGradFnF64)] = TWO_ARG_FUNCTIONS_F64;

pub fn function_resolver(name: &str) -> Option<*const u8> {
    let (base_name, is_tangent, real_type) = parse_function_name(name);

    let mut addr: *const u8 =
        resolve_by_real_type(base_name, is_tangent, &real_type).unwrap_or(std::ptr::null());

    // try f64 as a fallback when no explicit suffix is given
    if addr.is_null() && !matches!(real_type, RealType::F64) {
        addr =
            resolve_by_real_type(base_name, is_tangent, &RealType::F64).unwrap_or(std::ptr::null());
    }

    // include a libc lookup
    if addr.is_null() {
        addr = lookup_with_dlsym(name).unwrap_or(std::ptr::null());
    }

    if addr.is_null() {
        None
    } else {
        Some(addr)
    }
}

/// taken from https://github.com/bytecodealliance/wasmtime/blob/ee275a899a47adb14031aebc660580378cc2dc06/cranelift/jit/src/backend.rs#L636C1-L677C2
/// Apache License 2.0, see https://github.com/bytecodealliance/wasmtime/blob/ee275a899a47adb14031aebc660580378cc2dc06/LICENSE#L1
#[cfg(all(not(target_os = "windows"), not(target_arch = "wasm32"),))]
fn lookup_with_dlsym(name: &str) -> Option<*const u8> {
    let c_str = std::ffi::CString::new(name).unwrap();
    let c_str_ptr = c_str.as_ptr();
    let sym = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c_str_ptr) };
    if sym.is_null() {
        None
    } else {
        Some(sym as *const u8)
    }
}

#[cfg(target_arch = "wasm32")]
fn lookup_with_dlsym(_name: &str) -> Option<*const u8> {
    // no-op, as we don't need to look up symbols in wasm
    None
}

/// taken from https://github.com/bytecodealliance/wasmtime/blob/ee275a899a47adb14031aebc660580378cc2dc06/cranelift/jit/src/backend.rs#L636C1-L677C2
/// Apache License 2.0, see https://github.com/bytecodealliance/wasmtime/blob/ee275a899a47adb14031aebc660580378cc2dc06/LICENSE#L1
#[cfg(all(target_os = "windows", not(target_arch = "wasm32"),))]
fn lookup_with_dlsym(name: &str) -> Option<*const u8> {
    use std::os::windows::io::RawHandle;
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader;

    const UCRTBASE: &[u8] = b"ucrtbase.dll\0";

    let c_str = std::ffi::CString::new(name).unwrap();
    let c_str_ptr = c_str.as_ptr();

    unsafe {
        let handles = [
            // try to find the searched symbol in the currently running executable
            std::ptr::null_mut(),
            // try to find the searched symbol in local c runtime
            LibraryLoader::GetModuleHandleA(UCRTBASE.as_ptr()) as RawHandle,
        ];

        for handle in &handles {
            let addr = LibraryLoader::GetProcAddress(*handle as HMODULE, c_str_ptr.cast());
            match addr {
                None => continue,
                Some(addr) => return Some(addr as *const u8),
            }
        }

        None
    }
}

pub fn function_num_args(name: &str, is_tangent: bool) -> Option<usize> {
    let (base_name, _, _) = parse_function_name(name);
    let multiplier = if is_tangent { 2 } else { 1 };

    if FUNCTIONS_F64.iter().any(|(n, _, _)| n == &base_name)
        || FUNCTIONS_F32.iter().any(|(n, _, _)| n == &base_name)
    {
        return Some(multiplier);
    }

    if TWO_ARG_FUNCTIONS_F64
        .iter()
        .any(|(n, _, _)| n == &base_name)
        || TWO_ARG_FUNCTIONS_F32
            .iter()
            .any(|(n, _, _)| n == &base_name)
    {
        return Some(2 * multiplier);
    }

    None
}

// Explicit f64 versions
extern "C" fn sin_f64(x: f64) -> f64 {
    x.sin()
}
extern "C" fn dsin_f64(x: f64, dx: f64) -> f64 {
    x.cos() * dx
}

extern "C" fn cos_f64(x: f64) -> f64 {
    x.cos()
}
extern "C" fn dcos_f64(x: f64, dx: f64) -> f64 {
    -x.sin() * dx
}

extern "C" fn tan_f64(x: f64) -> f64 {
    x.tan()
}
extern "C" fn dtan_f64(x: f64, dx: f64) -> f64 {
    x.cos().powi(-2) * dx
}

extern "C" fn exp_f64(x: f64) -> f64 {
    x.exp()
}
extern "C" fn dexp_f64(x: f64, dx: f64) -> f64 {
    x.exp() * dx
}

extern "C" fn log_f64(x: f64) -> f64 {
    x.ln()
}
extern "C" fn dlog_f64(x: f64, dx: f64) -> f64 {
    dx / x
}

extern "C" fn log10_f64(x: f64) -> f64 {
    x.log10()
}
extern "C" fn dlog10_f64(x: f64, dx: f64) -> f64 {
    dx / (x * 10.0_f64.ln())
}

extern "C" fn sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}
extern "C" fn dsqrt_f64(x: f64, dx: f64) -> f64 {
    0.5 * dx / x.sqrt()
}

extern "C" fn abs_f64(x: f64) -> f64 {
    x.abs()
}
extern "C" fn dabs_f64(x: f64, dx: f64) -> f64 {
    if x > 0.0 {
        dx
    } else {
        -dx
    }
}

extern "C" fn copysign_f64(x: f64, y: f64) -> f64 {
    x.copysign(y)
}
extern "C" fn dcopysign_f64(_x: f64, dx: f64, y: f64, _dy: f64) -> f64 {
    dx.copysign(y)
}

extern "C" fn pow_f64(x: f64, y: f64) -> f64 {
    x.powf(y)
}
extern "C" fn dpow_f64(x: f64, dx: f64, y: f64, dy: f64) -> f64 {
    x.powf(y - 1.0) * (y * dx + x * dx.ln() * dy)
}

extern "C" fn min_f64(x: f64, y: f64) -> f64 {
    x.min(y)
}
extern "C" fn dmin_f64(x: f64, dx: f64, y: f64, dy: f64) -> f64 {
    if x < y {
        dx
    } else {
        dy
    }
}

extern "C" fn max_f64(x: f64, y: f64) -> f64 {
    x.max(y)
}
extern "C" fn dmax_f64(x: f64, dx: f64, y: f64, dy: f64) -> f64 {
    if x > y {
        dx
    } else {
        dy
    }
}

extern "C" fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
extern "C" fn dsigmoid_f64(x: f64, dx: f64) -> f64 {
    dx / (2.0 * x.cosh() + 2.0)
}

extern "C" fn arcsinh_f64(x: f64) -> f64 {
    x.asinh()
}
extern "C" fn darcsinh_f64(x: f64, dx: f64) -> f64 {
    dx / (x.powi(2) + 1.0).sqrt()
}

extern "C" fn arccosh_f64(x: f64) -> f64 {
    x.acosh()
}
extern "C" fn darccosh_f64(x: f64, dx: f64) -> f64 {
    dx / ((x - 1.0).sqrt() * (x + 1.0).sqrt())
}

extern "C" fn heaviside_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}
extern "C" fn dheaviside_f64(_x: f64, _dx: f64) -> f64 {
    0.0
}

extern "C" fn tanh_f64(x: f64) -> f64 {
    x.tanh()
}
extern "C" fn dtanh_f64(x: f64, dx: f64) -> f64 {
    dx / x.cosh().powi(2)
}

extern "C" fn sinh_f64(x: f64) -> f64 {
    x.sinh()
}
extern "C" fn dsinh_f64(x: f64, dx: f64) -> f64 {
    dx * x.cosh()
}

extern "C" fn cosh_f64(x: f64) -> f64 {
    x.cosh()
}
extern "C" fn dcosh_f64(x: f64, dx: f64) -> f64 {
    dx * x.sinh()
}

// Explicit f32 versions
extern "C" fn sin_f32(x: f32) -> f32 {
    x.sin()
}
extern "C" fn dsin_f32(x: f32, dx: f32) -> f32 {
    x.cos() * dx
}

extern "C" fn cos_f32(x: f32) -> f32 {
    x.cos()
}
extern "C" fn dcos_f32(x: f32, dx: f32) -> f32 {
    -x.sin() * dx
}

extern "C" fn tan_f32(x: f32) -> f32 {
    x.tan()
}
extern "C" fn dtan_f32(x: f32, dx: f32) -> f32 {
    x.cos().powi(-2) * dx
}

extern "C" fn exp_f32(x: f32) -> f32 {
    x.exp()
}
extern "C" fn dexp_f32(x: f32, dx: f32) -> f32 {
    x.exp() * dx
}

extern "C" fn log_f32(x: f32) -> f32 {
    x.ln()
}
extern "C" fn dlog_f32(x: f32, dx: f32) -> f32 {
    dx / x
}

extern "C" fn log10_f32(x: f32) -> f32 {
    x.log10()
}
extern "C" fn dlog10_f32(x: f32, dx: f32) -> f32 {
    dx / (x * 10.0_f32.ln())
}

extern "C" fn sqrt_f32(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn dsqrt_f32(x: f32, dx: f32) -> f32 {
    0.5 * dx / x.sqrt()
}

extern "C" fn abs_f32(x: f32) -> f32 {
    x.abs()
}
extern "C" fn dabs_f32(x: f32, dx: f32) -> f32 {
    if x > 0.0 {
        dx
    } else {
        -dx
    }
}

extern "C" fn copysign_f32(x: f32, y: f32) -> f32 {
    x.copysign(y)
}
extern "C" fn dcopysign_f32(_x: f32, dx: f32, y: f32, _dy: f32) -> f32 {
    dx.copysign(y)
}

extern "C" fn pow_f32(x: f32, y: f32) -> f32 {
    x.powf(y)
}
extern "C" fn dpow_f32(x: f32, dx: f32, y: f32, dy: f32) -> f32 {
    x.powf(y - 1.0) * (y * dx + x * dx.ln() * dy)
}

extern "C" fn min_f32(x: f32, y: f32) -> f32 {
    x.min(y)
}
extern "C" fn dmin_f32(x: f32, dx: f32, y: f32, dy: f32) -> f32 {
    if x < y {
        dx
    } else {
        dy
    }
}

extern "C" fn max_f32(x: f32, y: f32) -> f32 {
    x.max(y)
}
extern "C" fn dmax_f32(x: f32, dx: f32, y: f32, dy: f32) -> f32 {
    if x > y {
        dx
    } else {
        dy
    }
}

extern "C" fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
extern "C" fn dsigmoid_f32(x: f32, dx: f32) -> f32 {
    dx / (2.0 * x.cosh() + 2.0)
}

extern "C" fn arcsinh_f32(x: f32) -> f32 {
    x.asinh()
}
extern "C" fn darcsinh_f32(x: f32, dx: f32) -> f32 {
    dx / (x.powi(2) + 1.0).sqrt()
}

extern "C" fn arccosh_f32(x: f32) -> f32 {
    x.acosh()
}
extern "C" fn darccosh_f32(x: f32, dx: f32) -> f32 {
    dx / ((x - 1.0).sqrt() * (x + 1.0).sqrt())
}

extern "C" fn heaviside_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}
extern "C" fn dheaviside_f32(_x: f32, _dx: f32) -> f32 {
    0.0
}

extern "C" fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}
extern "C" fn dtanh_f32(x: f32, dx: f32) -> f32 {
    dx / x.cosh().powi(2)
}

extern "C" fn sinh_f32(x: f32) -> f32 {
    x.sinh()
}
extern "C" fn dsinh_f32(x: f32, dx: f32) -> f32 {
    dx * x.cosh()
}

extern "C" fn cosh_f32(x: f32) -> f32 {
    x.cosh()
}
extern "C" fn dcosh_f32(x: f32, dx: f32) -> f32 {
    dx * x.sinh()
}
