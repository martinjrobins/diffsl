#![allow(clippy::type_complexity)]
pub const FUNCTIONS: &[(
    &str,
    extern "C" fn(f64) -> f64,
    extern "C" fn(f64, f64) -> f64,
)] = &[
    ("sin", sin, dsin),
    ("cos", cos, dcos),
    ("tan", tan, dtan),
    ("exp", exp, dexp),
    ("log", log, dlog),
    ("log10", log10, dlog10),
    ("sqrt", sqrt, dsqrt),
    ("abs", abs, dabs),
    ("sigmoid", sigmoid, dsigmoid),
    ("arcsinh", arcsinh, darcsinh),
    ("arccosh", arccosh, darccosh),
    ("heaviside", heaviside, dheaviside),
    ("tanh", tanh, dtanh),
    ("sinh", sinh, dsinh),
    ("cosh", cosh, dcosh),
];

pub const TWO_ARG_FUNCTIONS: &[(
    &str,
    extern "C" fn(f64, f64) -> f64,
    extern "C" fn(f64, f64, f64, f64) -> f64,
)] = &[
    ("copysign", copysign, dcopysign),
    ("pow", pow, dpow),
    ("min", min, dmin),
    ("max", max, dmax),
];

pub fn function_resolver(name: &str) -> Option<*const u8> {
    let name = name.strip_prefix("_").unwrap_or(name);
    let mut addr: *const u8 = std::ptr::null();
    for func in crate::execution::functions::FUNCTIONS.iter() {
        if func.0 == name {
            addr = func.1 as *const u8;
        }
        if format!("{}__tangent__", func.0) == name {
            addr = func.2 as *const u8;
        }
    }
    for func in crate::execution::functions::TWO_ARG_FUNCTIONS.iter() {
        if func.0 == name {
            addr = func.1 as *const u8;
        }
        if format!("{}__tangent__", func.0) == name {
            addr = func.2 as *const u8;
        }
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
    let one = FUNCTIONS.iter().find(|(n, _, _)| n == &name);
    let multiplier = if is_tangent { 2 } else { 1 };
    if one.is_some() {
        return Some(multiplier);
    }
    let two = TWO_ARG_FUNCTIONS.iter().find(|(n, _, _)| n == &name);
    if two.is_some() {
        return Some(2 * multiplier);
    }
    None
}

extern "C" fn sin(x: f64) -> f64 {
    x.sin()
}

extern "C" fn dsin(x: f64, dx: f64) -> f64 {
    x.cos() * dx
}

extern "C" fn cos(x: f64) -> f64 {
    x.cos()
}

extern "C" fn dcos(x: f64, dx: f64) -> f64 {
    -x.sin() * dx
}

extern "C" fn tan(x: f64) -> f64 {
    x.tan()
}

extern "C" fn dtan(x: f64, dx: f64) -> f64 {
    let sec = x.cos().powi(-2);
    sec * dx
}

extern "C" fn exp(x: f64) -> f64 {
    x.exp()
}

extern "C" fn dexp(x: f64, dx: f64) -> f64 {
    x.exp() * dx
}

extern "C" fn log(x: f64) -> f64 {
    x.ln()
}

extern "C" fn dlog(x: f64, dx: f64) -> f64 {
    dx / x
}

extern "C" fn log10(x: f64) -> f64 {
    x.log10()
}

extern "C" fn dlog10(x: f64, dx: f64) -> f64 {
    dx / (x * 10.0_f64.ln())
}

extern "C" fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

extern "C" fn dsqrt(x: f64, dx: f64) -> f64 {
    0.5 * dx / x.sqrt()
}

extern "C" fn abs(x: f64) -> f64 {
    x.abs()
}

extern "C" fn dabs(x: f64, dx: f64) -> f64 {
    if x > 0.0 {
        dx
    } else {
        -dx
    }
}

extern "C" fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

// todo: this is not correct if b(x) == 0
extern "C" fn dcopysign(_x: f64, dx: f64, y: f64, _dy: f64) -> f64 {
    dx.copysign(y)
}

extern "C" fn pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

// d/dx(f(x)^g(x)) = f(x)^(g(x) - 1) (g(x) f'(x) + f(x) log(f(x)) g'(x))
extern "C" fn dpow(x: f64, dx: f64, y: f64, dy: f64) -> f64 {
    x.powf(y - 1.0) * (y * dx + x * dx.ln() * dy)
}

extern "C" fn min(x: f64, y: f64) -> f64 {
    x.min(y)
}

extern "C" fn dmin(x: f64, dx: f64, y: f64, dy: f64) -> f64 {
    if x < y {
        dx
    } else {
        dy
    }
}

extern "C" fn max(x: f64, y: f64) -> f64 {
    x.max(y)
}

extern "C" fn dmax(x: f64, dx: f64, y: f64, dy: f64) -> f64 {
    if x > y {
        dx
    } else {
        dy
    }
}

extern "C" fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// (f'(x))/(2 cosh(f(x)) + 2)
extern "C" fn dsigmoid(x: f64, dx: f64) -> f64 {
    let cosh = x.cosh();
    dx / (2.0 * cosh + 2.0)
}

extern "C" fn arcsinh(x: f64) -> f64 {
    x.asinh()
}

// d/dx(sinh^(-1)(f(x))) = (f'(x))/sqrt(f(x)^2 + 1)
extern "C" fn darcsinh(x: f64, dx: f64) -> f64 {
    dx / (x.powi(2) + 1.0).sqrt()
}

extern "C" fn arccosh(x: f64) -> f64 {
    x.acosh()
}

// d/dx(cosh^(-1)(f(x))) = (f'(x))/(sqrt(f(x) - 1) sqrt(f(x) + 1))
extern "C" fn darccosh(x: f64, dx: f64) -> f64 {
    dx / ((x - 1.0).sqrt() * (x + 1.0).sqrt())
}

extern "C" fn heaviside(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

// todo: not correct at a(x) == 0
extern "C" fn dheaviside(_x: f64, _dx: f64) -> f64 {
    0.0
}

extern "C" fn tanh(x: f64) -> f64 {
    x.tanh()
}

// (f'(x))/(cosh^2(f(x)))
extern "C" fn dtanh(x: f64, dx: f64) -> f64 {
    let cosh = x.cosh();
    dx / cosh.powi(2)
}

extern "C" fn sinh(x: f64) -> f64 {
    x.sinh()
}

// d/dx(sinh(f(x))) = f'(x) cosh(f(x))
extern "C" fn dsinh(x: f64, dx: f64) -> f64 {
    dx * x.cosh()
}

extern "C" fn cosh(x: f64) -> f64 {
    x.cosh()
}

// d/dx(cosh(f(x))) = f'(x) sinh(f(x))
extern "C" fn dcosh(x: f64, dx: f64) -> f64 {
    dx * x.sinh()
}
