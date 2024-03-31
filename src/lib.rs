//! ## DiffSL Language
//!
//! The DSL is designed to be an easy and flexible language for specifying
//! DAE systems and is based on the idea that a DAE system can be specified by a set
//! of equations of the form:
//!
//! $$
//! M(t) \frac{d\mathbf{u}}{dt} = F(\mathbf{u}, t)
//! $$
//!
//! where $\mathbf{u}$ is the vector of state variables and $t$ is the time. The DSL
//! allows the user to specify the state vector $\mathbf{u}$ and the RHS function $F$.
//! Optionally, the user can also define the derivative of the state vector $d\mathbf{u}/dt$
//! and the mass matrix $M$ as a function of $d\mathbf{u}/dt$ (note that this function should be linear!).
//! The user is also free to define an an arbitrary number of intermediate
//! scalars and vectors of the users that are required to calculate $F$ and $M$.
//!
//! ### Defining variables
//!
//! The DSL allows the user to define scalars, vectors, and dense/sparse/diagonal n-dimensional tensors.
//! You can optionally label the elements of a vector or tensor for later use.
//!
//! For example, to define a scalar variable `k` with value `1.0`, we write:
//!
//! ```
//! k { 1.0 }
//! ```
//!
//! To define a vector variable `v` with 3 elements that are labelled, we write:
//!
//! ```
//! v_i {
//!  x = 1.0,
//!  y = 2.0,
//!  z = 3.0,
//! }
//! ````
//!
//! The subscript `_i` indicates that this is a 1D vector, and `x`, `y`, and `z` are defined as labels to the 3 elements of the vector.
//! Later in the code, we could refer to either the whole vector `v_i` or to the individual elements `x`, `y`, and `z`, which are scalars.
//!
//! To define dense 2x3 matrix `A` with all elements set to `1.0`, we write:
//!
//! ```
//! A_ij {
//!  (0:2, 0:3) = 1.0,
//! }
//! ```
//!
//! Note the two subscript to indicate that this is a 2D tensor. The size of the tensor is given in the brackets, and the elements are set to `1.0`.
//! If we have additional rows, we can add them as follows:
//!
//! ```
//! A_ij {
//!  (0:2, 0:3) = 1.0,
//!  (3:4, 0:3) = 2.0,
//! }
//! ```
//!
//! We can define a sparse matrix `B` by specifying the non-zero elements:
//!
//! ```
//! B_ij {
//!  (0, 0) = 1.0,
//!  (0, 1) = 2.0,
//!  (1, 1) = 3.0,
//! }
//! ```
//!
//! We can also define a diagonal identity matrix `I` by specifying the diagonal elements using a different range syntax:
//!
//! ```
//! I_ij {
//!  (0..2, 0..2) = 1.0,
//! }
//! ```
//!
//! ### Operations
//!
//! We can use standard algebraic operations on variables. To refer to previously defined variables,
//! we use the variable name, making sure to use the correct subscript if it is a vector or tensor.
//!
//! For example, to define a scalar variable `a` as the sum of two other scalar variables `b` and `c`, we write:
//!
//! ```
//! a { b + c }
//! ```
//!
//! To define a vector variable `v` as the sum of two other vector variables `u` and `w`, we write:
//!
//! ```
//! v_i { u_i + w_i }
//! ```
//!
//! The indexing can be used to perform translations on tensors, for example the following will define a new tensor $C$ that is the sum of $A$ and $B^T$:
//!
//! ```
//! C_ij { A_ij + B_ji }
//! ```
//!
//! Tensor indexing notation can also matrix-vector multiplications and any other contraction operations. Any indices that do not appear in the output will be summed over.
//! For example, the following will define a new vector $v$ that is the result of a matrix-vector multiplication:
//!
//! ```
//! v_i { A_ij * u_j }
//! ```
//!
//! ### Specifying inputs
//!
//! We can override the values of any scalar variables by specifying them as input variables.
//! To do this, we add a line at the top
//! of the code to specify that these are input variables:
//!
//! ```
//! in = [k]
//! k { 1.0 }
//! ```
//!
//! ### Defining state variables
//!
//! The primary goal of the DSL is to define a set of differential equations of a system of state variables.
//! To define the state variables, we create a special vector variable `u_i` which corresponds to the state variables $\mathbf{u}$.
//!
//! The values that we use for `u_i` are the initial values of the state variables at $t=0$.
//!
//! ```
//! u_i {
//!   x = 1,
//!   y = 0,
//!   z = 0,
//! }
//! ```
//!
//! We can optionally define the time derivatives of the state variables, $\mathbf{\dot{u}}$ as well:
//!
//! ```
//! dudt_i {
//!   dxdt = 1,
//!   dydt = 0,
//!   dzdt = 0,
//! }
//! ```
//!
//! Here the initial values of the time derivatives are given, these are typically  used as a starting point to calculate a set of consistent
//! initial values for the state variables.
//!
//! Note that there is no need to define `dudt` if you do not define a mass matrix $M$.
//!
//! ### Defining the ODE system equations
//!
//! Recall that the DAE system is defined by the equations:
//!
//! $$
//! M(t) \frac{d\mathbf{u}}{dt} = F(\mathbf{u}, t)
//! $$
//!
//! We now define the equations $F$ and $M$ that we want to solve, using the
//! variables that we have defined earlier. We do this by defining a vector variable `F_i` that
//! corresponds to the RHS of the equations.
//!
//! For example, to define a simple system of ODEs:
//!
//! $$
//! \begin{align*}
//!  \frac{dx}{dt} &= y \\
//!  \frac{dy}{dt} &= -x \\
//!  x(0) &= 1 \\
//!  y(0) &= 0 \\
//! \end{align*}
//! $$
//!
//! We write:
//!
//! ```
//! u_i {
//!  x = 1,
//!  y = 0,
//! }
//! F_i {
//!   y,
//!  -x,
//! }
//! ```
//!
//! We can also define a mass matrix $M$ by defining a vector variable `M_i`. This is optional, and if not defined, the mass matrix is assumed to be the identity matrix.
//!
//! For example, lets define a simple DAE system using a singular mass matrix with a zero on the diagonal:
//!
//! $$
//! \begin{align*}
//!  \frac{dx}{dt} &= x \\
//!  0 &= y-x \\
//!  x(0) &= 1 \\
//!  y(0) &= 0 \\
//! \end{align*}
//! $$
//!
//! We write:
//!
//! ```
//! u_i {
//!  x = 1,
//!  y = 0,
//! }
//! dudt_i {
//!  dxdt = 0,
//!  dydt = 1,
//! }
//! M_i {
//!  dxdt,
//!  0,
//! }
//! F_i {
//!  x,
//!  y-x,
//! }
//! ```
//!
//! ### Specifying outputs
//!
//! Finally, we specify the outputs of the system. These might be the state
//! variables themselves, or they might be other variables that are calculated from
//! the state variables. Here we specify that we want to output the state variables
//! `x` and `y`:
//!
//! ```
//! out_i {
//!   x,
//!   y,
//! }
//! ```
//!
//! ### Required variables
//!
//! The DSL allows the user to specify an arbitrary number of intermediate variables, but certain variables are required to be defined. These are:
//!
//! * `u_i` - the state variables
//! * `F_i` - the vector $F(\mathbf{u}, t)$
//! * `out_i` - the output variables
//!
//! ### Predefined variables
//!
//! The only predefined variable is the scalar `t` which is the current time, this allows the equations to be written as functions of time. For example
//!
//! ```
//! F_i {
//!   k1 * t + sin(t)
//! }
//! ```
//!
//! ### Mathematical functions
//!
//! The DSL supports the following mathematical functions:
//!
//! * `pow(x, y)` - x raised to the power of y
//! * `sin(x)` - sine of x
//! * `cos(x)` - cosine of x
//! * `tan(x)` - tangent of x
//! * `exp(x)` - exponential of x
//! * `log(x)` - natural logarithm of x
//! * `sqrt(x)` - square root of x
//! * `abs(x)` - absolute value of x
//! * `sigmoid(x)` - sigmoid function of x

use anyhow::{anyhow, Result};
use continuous::ModelInfo;
use discretise::DiscreteModel;
use execution::Compiler;
use parser::{parse_ds_string, parse_ms_string};
use std::{ffi::OsStr, path::Path};

extern crate pest;
#[macro_use]
extern crate pest_derive;

pub mod ast;
pub mod continuous;
pub mod discretise;
pub mod execution;
pub mod parser;
pub mod utils;

#[cfg(feature = "inkwell-100")]
pub extern crate inkwell_100 as inkwell;
#[cfg(feature = "inkwell-110")]
pub extern crate inkwell_110 as inkwell;
#[cfg(feature = "inkwell-120")]
pub extern crate inkwell_120 as inkwell;
#[cfg(feature = "inkwell-130")]
pub extern crate inkwell_130 as inkwell;
#[cfg(feature = "inkwell-140")]
pub extern crate inkwell_140 as inkwell;
#[cfg(feature = "inkwell-150")]
pub extern crate inkwell_150 as inkwell;
#[cfg(feature = "inkwell-160")]
pub extern crate inkwell_160 as inkwell;
#[cfg(feature = "inkwell-170")]
pub extern crate inkwell_170 as inkwell;
#[cfg(feature = "inkwell-40")]
pub extern crate inkwell_40 as inkwell;
#[cfg(feature = "inkwell-50")]
pub extern crate inkwell_50 as inkwell;
#[cfg(feature = "inkwell-60")]
pub extern crate inkwell_60 as inkwell;
#[cfg(feature = "inkwell-70")]
pub extern crate inkwell_70 as inkwell;
#[cfg(feature = "inkwell-80")]
pub extern crate inkwell_80 as inkwell;
#[cfg(feature = "inkwell-90")]
pub extern crate inkwell_90 as inkwell;

pub struct CompilerOptions {
    pub bitcode_only: bool,
    pub wasm: bool,
    pub standalone: bool,
}

pub fn compile(
    input: &str,
    out: Option<&str>,
    model: Option<&str>,
    options: CompilerOptions,
) -> Result<()> {
    let inputfile = Path::new(input);
    let is_discrete = inputfile
        .extension()
        .unwrap_or(OsStr::new(""))
        .to_str()
        .unwrap()
        == "ds";
    let is_continuous = inputfile
        .extension()
        .unwrap_or(OsStr::new(""))
        .to_str()
        .unwrap()
        == "cs";
    if !is_discrete && !is_continuous {
        panic!("Input file must have extension .ds or .cs");
    }
    let model_name = if is_continuous {
        if let Some(model_name) = model {
            model_name
        } else {
            return Err(anyhow!(
                "Model name must be specified for continuous models"
            ));
        }
    } else {
        inputfile.file_stem().unwrap().to_str().unwrap()
    };
    let out = out.unwrap_or("out");
    let text = std::fs::read_to_string(inputfile)?;
    compile_text(text.as_str(), out, model_name, options, is_discrete)
}

pub fn compile_text(
    text: &str,
    out: &str,
    model_name: &str,
    options: CompilerOptions,
    is_discrete: bool,
) -> Result<()> {
    let is_continuous = !is_discrete;

    let continuous_ast = if is_continuous {
        Some(parse_ms_string(text)?)
    } else {
        None
    };

    let discrete_ast = if is_discrete {
        Some(parse_ds_string(text)?)
    } else {
        None
    };

    let continuous_model_info = if let Some(ast) = &continuous_ast {
        let model_info = ModelInfo::build(model_name, ast).map_err(|e| anyhow!("{}", e))?;
        if !model_info.errors.is_empty() {
            let error_text = model_info.errors.iter().fold(String::new(), |acc, error| {
                format!("{}\n{}", acc, error.as_error_message(text))
            });
            return Err(anyhow!(error_text));
        }
        Some(model_info)
    } else {
        None
    };

    let discrete_model = if let Some(model_info) = &continuous_model_info {
        let model = DiscreteModel::from(model_info);
        model
    } else if let Some(ast) = &discrete_ast {
        match DiscreteModel::build(model_name, ast) {
            Ok(model) => model,
            Err(e) => {
                return Err(anyhow!(e.as_error_message(text)));
            }
        }
    } else {
        panic!("No model found");
    };
    let compiler = Compiler::from_discrete_model(&discrete_model, out)?;

    if options.bitcode_only {
        return Ok(());
    }

    compiler.compile(options.standalone, options.wasm)
}

#[cfg(test)]
mod tests {
    use crate::{
        continuous::ModelInfo,
        parser::{parse_ds_string, parse_ms_string},
    };
    use approx::assert_relative_eq;

    use super::*;

    fn ds_example_compiler(example: &str) -> Compiler {
        let text = std::fs::read_to_string(format!("examples/{}.ds", example)).unwrap();
        let model = parse_ds_string(text.as_str()).unwrap();
        let model = DiscreteModel::build(example, &model)
            .unwrap_or_else(|e| panic!("{}", e.as_error_message(text.as_str())));
        let out = format!("test_output/lib_examples_{}", example);
        Compiler::from_discrete_model(&model, out.as_str()).unwrap()
    }

    #[test]
    fn test_logistic_ds_example() {
        let compiler = ds_example_compiler("logistic");
        let r = 0.5;
        let k = 0.5;
        let y = 0.5;
        let dydt = r * y * (1. - y / k);
        let z = 2. * y;
        let dzdt = 2. * dydt;
        let inputs = vec![r, k];
        let mut u0 = vec![y, z];
        let mut data = compiler.get_new_data();
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice());
        compiler.set_u0(u0.as_mut_slice(), data.as_mut_slice());

        u0 = vec![y, z];
        let up0 = vec![dydt, dzdt];
        let mut res = vec![1., 1.];

        compiler.rhs(0., u0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        let expected_value = vec![dydt, 2.0 * y - z];
        assert_relative_eq!(res.as_slice(), expected_value.as_slice());

        compiler.mass(0., up0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        let expected_value = vec![dydt, 0.];
        assert_relative_eq!(res.as_slice(), expected_value.as_slice());
    }

    #[test]
    fn test_object_file() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), z(t)) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
            z = 2 * y
        }
        ";
        let models = parse_ms_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete_model = DiscreteModel::from(&model_info);
        let object =
            Compiler::from_discrete_model(&discrete_model, "test_output/lib_test_object_file")
                .unwrap();
        let path = Path::new("main.o");
        object.write_object_file(path).unwrap();
    }
}
