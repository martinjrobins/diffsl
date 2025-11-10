use std::fmt::Debug;

use approx::AbsDiffEq;
use num_traits::FromPrimitive;

pub enum RealType {
    F32,
    F64,
}

impl RealType {
    pub fn as_str(&self) -> &str {
        match self {
            RealType::F32 => "f32",
            RealType::F64 => "f64",
        }
    }
}

pub trait Scalar:
    Copy + FromPrimitive + Debug + num_traits::Signed + AbsDiffEq<Epsilon: Clone> + Sync + 'static
{
    fn as_real_type() -> RealType;
}

impl Scalar for f64 {
    fn as_real_type() -> RealType {
        RealType::F64
    }
}

impl Scalar for f32 {
    fn as_real_type() -> RealType {
        RealType::F32
    }
}
