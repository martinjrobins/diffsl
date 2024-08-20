
#![allow(clippy::type_complexity)]
pub const FUNCTIONS: &[(&str, fn(&f64) -> f64, fn(&f64, &f64) -> f64)] = &[
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

pub const TWO_ARG_FUNCTIONS: &[(&str, fn(&f64, &f64) -> f64, fn(&f64, &f64, &f64, &f64) -> f64)] = &[
    ("copysign", copysign, dcopysign),
    ("pow", pow, dpow),
    ("min", min, dmin),
    ("max", max, dmax),
];

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


pub fn sin(x: &f64) -> f64 {
    x.sin()
}

pub fn dsin(x: &f64, dx: &f64) -> f64 {
    x.cos() * dx
}

pub fn cos(x: &f64) -> f64 {
    x.cos()
}

pub fn dcos(x: &f64, dx: &f64) -> f64 {
    -x.sin() * dx
}

pub fn tan(x: &f64) -> f64 {
    x.tan()
}

pub fn dtan(x: &f64, dx: &f64) -> f64 {
    let sec = x.cos().powi(-2);
    sec * dx
}

pub fn exp(x: &f64) -> f64 {
    x.exp()
}

pub fn dexp(x: &f64, dx: &f64) -> f64 {
    x.exp() * dx
}

pub fn log(x: &f64) -> f64 {
    x.ln()
}

pub fn dlog(x: &f64, dx: &f64) -> f64 {
    dx / x
}

pub fn log10(x: &f64) -> f64 {
    x.log10()
}

pub fn dlog10(x: &f64, dx: &f64) -> f64 {
    dx / (x * 10.0_f64.ln())
}

pub fn sqrt(x: &f64) -> f64 {
    x.sqrt()
}

pub fn dsqrt(x: &f64, dx: &f64) -> f64 {
    0.5 * dx / x.sqrt()
}

pub fn abs(x: &f64) -> f64 {
    x.abs()
}

pub fn dabs(x: &f64, dx: &f64) -> f64 {
    if *x > 0.0 {
        *dx
    } else {
        -dx
    }
}

pub fn copysign(x: &f64, y: &f64) -> f64 {
    x.copysign(*y)
}

// todo: this is not correct if b(x) == 0
pub fn dcopysign(_x: &f64, dx: &f64, y: &f64, _dy: &f64) -> f64 {
    dx.copysign(*y)
}

pub fn pow(x: &f64, y: &f64) -> f64 {
    x.powf(*y)
}

// d/dx(f(x)^g(x)) = f(x)^(g(x) - 1) (g(x) f'(x) + f(x) log(f(x)) g'(x))
pub fn dpow(x: &f64, dx: &f64, y: &f64, dy: &f64) -> f64 {
    x.powf(y - 1.0) * (y * dx + x * dx.ln() * dy)
}

pub fn min(x: &f64, y: &f64) -> f64 {
    x.min(*y)
}

pub fn dmin(x: &f64, dx: &f64, y: &f64, dy: &f64) -> f64 {
    if *x < *y {
        *dx
    } else {
        *dy
    }
}

pub fn max(x: &f64, y: &f64) -> f64 {
    x.max(*y)
}

pub fn dmax(x: &f64, dx: &f64, y: &f64, dy: &f64) -> f64 {
    if *x > *y {
        *dx
    } else {
        *dy
    }
}

pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// (f'(x))/(2 cosh(f(x)) + 2)       
pub fn dsigmoid(x: &f64, dx: &f64) -> f64 {
    let cosh = x.cosh();
    dx / (2.0 * cosh + 2.0)
}

pub fn arcsinh(x: &f64) -> f64 {
    x.asinh()
}

// d/dx(sinh^(-1)(f(x))) = (f'(x))/sqrt(f(x)^2 + 1)
pub fn darcsinh(x: &f64, dx: &f64) -> f64 {
    dx / (x.powi(2) + 1.0).sqrt()
}

pub fn arccosh(x: &f64) -> f64 {
    x.acosh()
}

// d/dx(cosh^(-1)(f(x))) = (f'(x))/(sqrt(f(x) - 1) sqrt(f(x) + 1))
pub fn darccosh(x: &f64, dx: &f64) -> f64 {
    dx / ((x - 1.0).sqrt() * (x + 1.0).sqrt())
}

pub fn heaviside(x: &f64) -> f64 {
    if *x > 0.0 {
        1.0
    } else if *x < 0.0 {
        0.0
    } else {
        0.5
    }
}

// todo: not correct at a(x) == 0
pub fn dheaviside(_x: &f64, _dx: &f64) -> f64 {
    0.0
}

pub fn tanh(x: &f64) -> f64 {
    x.tanh()
}


// (f'(x))/(cosh^2(f(x)))
pub fn dtanh(x: &f64, dx: &f64) -> f64 {
    let cosh = x.cosh();
    dx / cosh.powi(2)
}

pub fn sinh(x: &f64) -> f64 {
    x.sinh()
}

// d/dx(sinh(f(x))) = f'(x) cosh(f(x))
pub fn dsinh(x: &f64, dx: &f64) -> f64 {
    dx * x.cosh()
}

pub fn cosh(x: &f64) -> f64 {
    x.cosh()
}

// d/dx(cosh(f(x))) = f'(x) sinh(f(x))
pub fn dcosh(x: &f64, dx: &f64) -> f64 {
    dx * x.sinh()
}

