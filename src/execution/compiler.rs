use crate::{
    discretise::DiscreteModel,
    execution::interface::{
        CalcOutFunc, GetDimsFunc, GetOutFunc, MassFunc, RhsFunc, SetIdFunc, SetInputsFunc,
        StopFunc, U0Func,
    },
    parser::parse_ds_string,
};

use super::{
    interface::{CalcOutGradientFunc, RhsGradientFunc, SetInputsGradientFunc, U0GradientFunc},
    module::CodegenModule,
};
use anyhow::{anyhow, Result};
use target_lexicon::Triple;
use uid::Id;

struct JitFunctions {
    set_u0: U0Func,
    rhs: RhsFunc,
    mass: MassFunc,
    calc_out: CalcOutFunc,
    calc_stop: StopFunc,
    set_id: SetIdFunc,
    get_dims: GetDimsFunc,
    set_inputs: SetInputsFunc,
    get_out: GetOutFunc,
}

struct JitGradFunctions {
    set_u0_grad: U0GradientFunc,
    rhs_grad: RhsGradientFunc,
    calc_out_grad: CalcOutGradientFunc,
    set_inputs_grad: SetInputsGradientFunc,
}

pub struct Compiler<M: CodegenModule> {
    module: M,
    jit_functions: JitFunctions,
    jit_grad_functions: JitGradFunctions,

    number_of_states: usize,
    number_of_parameters: usize,
    number_of_outputs: usize,
    has_mass: bool,
}

impl<M: CodegenModule> Compiler<M> {
    pub fn from_discrete_str(code: &str) -> Result<Self> {
        let uid = Id::<u32>::new();
        let name = format!("diffsl_{}", uid);
        let model = parse_ds_string(code).map_err(|e| anyhow!(e.to_string()))?;
        let model = DiscreteModel::build(name.as_str(), &model)
            .map_err(|e| anyhow!(e.as_error_message(code)))?;
        Self::from_discrete_model(&model)
    }

    pub fn from_discrete_model(model: &DiscreteModel) -> Result<Self> {
        let number_of_states = *model.state().shape().first().unwrap_or(&1);
        let input_names = model
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect::<Vec<_>>();
        let mut module = M::new(Triple::host(), model)?;
        let number_of_parameters = input_names.iter().fold(0, |acc, name| {
            acc + module.layout().get_data_length(name).unwrap()
        });
        let number_of_outputs = module.layout().get_data_length("out").unwrap();
        let has_mass = model.lhs().is_some();

        let set_u0 = module.compile_set_u0(model)?;
        let calc_stop = module.compile_calc_stop(model)?;
        let rhs = module.compile_rhs(model)?;
        let mass = module.compile_mass(model)?;
        let calc_out = module.compile_calc_out(model)?;
        let set_id = module.compile_set_id(model)?;
        let get_dims = module.compile_get_dims(model)?;
        let set_inputs = module.compile_set_inputs(model)?;
        let get_output = module.compile_get_tensor(model, "out")?;

        module.pre_autodiff_optimisation()?;

        let set_u0_grad = module.compile_set_u0_grad(&set_u0, model)?;
        let rhs_grad = module.compile_rhs_grad(&rhs, model)?;
        let calc_out_grad = module.compile_calc_out_grad(&calc_out, model)?;
        let set_inputs_grad = module.compile_set_inputs_grad(&set_inputs, model)?;

        module.post_autodiff_optimisation()?;

        let set_u0 = unsafe { std::mem::transmute::<*const u8, U0Func>(module.jit(set_u0)?) };
        let rhs = unsafe { std::mem::transmute::<*const u8, RhsFunc>(module.jit(rhs)?) };
        let mass = unsafe { std::mem::transmute::<*const u8, MassFunc>(module.jit(mass)?) };
        let calc_out =
            unsafe { std::mem::transmute::<*const u8, CalcOutFunc>(module.jit(calc_out)?) };
        let calc_stop =
            unsafe { std::mem::transmute::<*const u8, StopFunc>(module.jit(calc_stop)?) };
        let set_id = unsafe { std::mem::transmute::<*const u8, SetIdFunc>(module.jit(set_id)?) };
        let get_dims =
            unsafe { std::mem::transmute::<*const u8, GetDimsFunc>(module.jit(get_dims)?) };
        let set_inputs =
            unsafe { std::mem::transmute::<*const u8, SetInputsFunc>(module.jit(set_inputs)?) };
        let get_out =
            unsafe { std::mem::transmute::<*const u8, GetOutFunc>(module.jit(get_output)?) };

        let set_u0_grad =
            unsafe { std::mem::transmute::<*const u8, U0GradientFunc>(module.jit(set_u0_grad)?) };
        let rhs_grad =
            unsafe { std::mem::transmute::<*const u8, RhsGradientFunc>(module.jit(rhs_grad)?) };
        let calc_out_grad = unsafe {
            std::mem::transmute::<*const u8, CalcOutGradientFunc>(module.jit(calc_out_grad)?)
        };
        let set_inputs_grad = unsafe {
            std::mem::transmute::<*const u8, SetInputsGradientFunc>(module.jit(set_inputs_grad)?)
        };

        Ok(Self {
            module,
            jit_functions: JitFunctions {
                set_u0,
                rhs,
                mass,
                calc_out,
                calc_stop,
                set_id,
                get_dims,
                set_inputs,
                get_out,
            },
            jit_grad_functions: JitGradFunctions {
                set_u0_grad,
                rhs_grad,
                calc_out_grad,
                set_inputs_grad,
            },
            number_of_states,
            number_of_parameters,
            number_of_outputs,
            has_mass,
        })
    }

    pub fn get_tensor_data<'a>(&self, name: &str, data: &'a [f64]) -> Option<&'a [f64]> {
        let index = self.module.layout().get_data_index(name)?;
        let nnz = self.module.layout().get_data_length(name)?;
        Some(&data[index..index + nnz])
    }

    pub fn set_u0(&self, yy: &mut [f64], data: &mut [f64]) {
        if yy.len() != self.number_of_states {
            panic!(
                "Expected {} states, got {}",
                self.number_of_states,
                yy.len()
            );
        }
        unsafe { (self.jit_functions.set_u0)(yy.as_mut_ptr(), data.as_mut_ptr()) };
    }

    pub fn set_u0_grad(
        &self,
        yy: &mut [f64],
        dyy: &mut [f64],
        data: &mut [f64],
        ddata: &mut [f64],
    ) {
        if yy.len() != self.number_of_states {
            panic!(
                "Expected {} states, got {}",
                self.number_of_states,
                yy.len()
            );
        }
        if dyy.len() != self.number_of_states {
            panic!(
                "Expected {} states for dyy, got {}",
                self.number_of_states,
                dyy.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        if ddata.len() != self.data_len() {
            panic!(
                "Expected {} data for ddata, got {}",
                self.data_len(),
                ddata.len()
            );
        }
        unsafe {
            (self.jit_grad_functions.set_u0_grad)(
                yy.as_mut_ptr(),
                dyy.as_mut_ptr(),
                data.as_mut_ptr(),
                ddata.as_mut_ptr(),
            )
        };
    }

    pub fn calc_stop(&self, t: f64, yy: &[f64], data: &mut [f64], stop: &mut [f64]) {
        let (n_states, _, _, n_data, n_stop, _) = self.get_dims();
        if yy.len() != n_states {
            panic!("Expected {} states, got {}", n_states, yy.len());
        }
        if data.len() != n_data {
            panic!("Expected {} data, got {}", n_data, data.len());
        }
        if stop.len() != n_stop {
            panic!("Expected {} stop, got {}", n_stop, stop.len());
        }
        unsafe {
            (self.jit_functions.calc_stop)(t, yy.as_ptr(), data.as_mut_ptr(), stop.as_mut_ptr())
        };
    }

    pub fn rhs(&self, t: f64, yy: &[f64], data: &mut [f64], rr: &mut [f64]) {
        if yy.len() != self.number_of_states {
            panic!(
                "Expected {} states, got {}",
                self.number_of_states,
                yy.len()
            );
        }
        if rr.len() != self.number_of_states {
            panic!(
                "Expected {} residual states, got {}",
                self.number_of_states,
                rr.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        unsafe { (self.jit_functions.rhs)(t, yy.as_ptr(), data.as_mut_ptr(), rr.as_mut_ptr()) };
    }

    pub fn has_mass(&self) -> bool {
        self.has_mass
    }

    pub fn mass(&self, t: f64, yp: &[f64], data: &mut [f64], rr: &mut [f64]) {
        if !self.has_mass {
            panic!("Model does not have a mass function");
        }
        if yp.len() != self.number_of_states {
            panic!(
                "Expected {} states, got {}",
                self.number_of_states,
                yp.len()
            );
        }
        if rr.len() != self.number_of_states {
            panic!(
                "Expected {} residual states, got {}",
                self.number_of_states,
                rr.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        unsafe { (self.jit_functions.mass)(t, yp.as_ptr(), data.as_mut_ptr(), rr.as_mut_ptr()) };
    }

    pub fn data_len(&self) -> usize {
        self.module.layout().data().len()
    }

    pub fn get_new_data(&self) -> Vec<f64> {
        vec![0.; self.data_len()]
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rhs_grad(
        &self,
        t: f64,
        yy: &[f64],
        dyy: &[f64],
        data: &mut [f64],
        ddata: &mut [f64],
        rr: &mut [f64],
        drr: &mut [f64],
    ) {
        if yy.len() != self.number_of_states {
            panic!(
                "Expected {} states, got {}",
                self.number_of_states,
                yy.len()
            );
        }
        if rr.len() != self.number_of_states {
            panic!(
                "Expected {} residual states, got {}",
                self.number_of_states,
                rr.len()
            );
        }
        if dyy.len() != self.number_of_states {
            panic!(
                "Expected {} states for dyy, got {}",
                self.number_of_states,
                dyy.len()
            );
        }
        if drr.len() != self.number_of_states {
            panic!(
                "Expected {} residual states for drr, got {}",
                self.number_of_states,
                drr.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        if ddata.len() != self.data_len() {
            panic!(
                "Expected {} data for ddata, got {}",
                self.data_len(),
                ddata.len()
            );
        }
        unsafe {
            (self.jit_grad_functions.rhs_grad)(
                t,
                yy.as_ptr(),
                dyy.as_ptr(),
                data.as_mut_ptr(),
                ddata.as_mut_ptr(),
                rr.as_mut_ptr(),
                drr.as_mut_ptr(),
            )
        };
    }

    pub fn calc_out(&self, t: f64, yy: &[f64], data: &mut [f64]) {
        if yy.len() != self.number_of_states {
            panic!(
                "Expected {} states, got {}",
                self.number_of_states,
                yy.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        unsafe { (self.jit_functions.calc_out)(t, yy.as_ptr(), data.as_mut_ptr()) };
    }

    pub fn calc_out_grad(
        &self,
        t: f64,
        yy: &[f64],
        dyy: &[f64],
        data: &mut [f64],
        ddata: &mut [f64],
    ) {
        if yy.len() != self.number_of_states {
            panic!(
                "Expected {} states, got {}",
                self.number_of_states,
                yy.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        if dyy.len() != self.number_of_states {
            panic!(
                "Expected {} states for dyy, got {}",
                self.number_of_states,
                dyy.len()
            );
        }
        if ddata.len() != self.data_len() {
            panic!(
                "Expected {} data for ddata, got {}",
                self.data_len(),
                ddata.len()
            );
        }
        unsafe {
            (self.jit_grad_functions.calc_out_grad)(
                t,
                yy.as_ptr(),
                dyy.as_ptr(),
                data.as_mut_ptr(),
                ddata.as_mut_ptr(),
            )
        };
    }

    /// Get various dimensions of the model
    ///
    /// # Returns
    ///
    /// A tuple of the form `(n_states, n_inputs, n_outputs, n_data, n_stop, has_mass)`
    pub fn get_dims(&self) -> (usize, usize, usize, usize, usize, bool) {
        let mut n_states = 0u32;
        let mut n_inputs = 0u32;
        let mut n_outputs = 0u32;
        let mut n_data = 0u32;
        let mut n_stop = 0u32;
        let mut has_mass = 0u32;
        unsafe {
            (self.jit_functions.get_dims)(
                &mut n_states,
                &mut n_inputs,
                &mut n_outputs,
                &mut n_data,
                &mut n_stop,
                &mut has_mass,
            )
        };
        (
            n_states as usize,
            n_inputs as usize,
            n_outputs as usize,
            n_data as usize,
            n_stop as usize,
            has_mass != 0,
        )
    }

    pub fn set_inputs(&self, inputs: &[f64], data: &mut [f64]) {
        let (_, n_inputs, _, _, _, _) = self.get_dims();
        if n_inputs != inputs.len() {
            panic!("Expected {} inputs, got {}", n_inputs, inputs.len());
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        unsafe { (self.jit_functions.set_inputs)(inputs.as_ptr(), data.as_mut_ptr()) };
    }

    pub fn set_inputs_grad(
        &self,
        inputs: &[f64],
        dinputs: &[f64],
        data: &mut [f64],
        ddata: &mut [f64],
    ) {
        let (_, n_inputs, _, _, _, _) = self.get_dims();
        if n_inputs != inputs.len() {
            panic!("Expected {} inputs, got {}", n_inputs, inputs.len());
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        if dinputs.len() != n_inputs {
            panic!(
                "Expected {} inputs for dinputs, got {}",
                n_inputs,
                dinputs.len()
            );
        }
        if ddata.len() != self.data_len() {
            panic!(
                "Expected {} data for ddata, got {}",
                self.data_len(),
                ddata.len()
            );
        }
        unsafe {
            (self.jit_grad_functions.set_inputs_grad)(
                inputs.as_ptr(),
                dinputs.as_ptr(),
                data.as_mut_ptr(),
                ddata.as_mut_ptr(),
            )
        };
    }

    pub fn get_out(&self, data: &[f64]) -> &[f64] {
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        let (_, _, n_outputs, _, _, _) = self.get_dims();
        let mut tensor_data_ptr: *mut f64 = std::ptr::null_mut();
        let mut tensor_data_len = 0u32;
        let tensor_data_ptr_ptr: *mut *mut f64 = &mut tensor_data_ptr;
        let tensor_data_len_ptr: *mut u32 = &mut tensor_data_len;
        unsafe {
            (self.jit_functions.get_out)(data.as_ptr(), tensor_data_ptr_ptr, tensor_data_len_ptr)
        };
        assert!(tensor_data_len as usize == n_outputs);
        unsafe { std::slice::from_raw_parts(tensor_data_ptr, tensor_data_len as usize) }
    }

    pub fn set_id(&self, id: &mut [f64]) {
        let (n_states, _, _, _, _, _) = self.get_dims();
        if n_states != id.len() {
            panic!("Expected {} states, got {}", n_states, id.len());
        }
        unsafe { (self.jit_functions.set_id)(id.as_mut_ptr()) };
    }

    pub fn number_of_states(&self) -> usize {
        self.number_of_states
    }
    pub fn number_of_parameters(&self) -> usize {
        self.number_of_parameters
    }

    pub fn number_of_outputs(&self) -> usize {
        self.number_of_outputs
    }
    pub fn module(&self) -> &M {
        &self.module
    }
}

#[cfg(test)]
mod tests {
    use crate::{parser::parse_ds_string, CraneliftModule};
    use approx::assert_relative_eq;

    use super::*;

    #[cfg(feature = "llvm")]
    #[test]
    fn test_from_discrete_str_llvm() {
        use crate::execution::llvm::codegen::LlvmModule;
        let text = "
        u { y = 1 }
        F { -y }
        out { y }
        ";
        let compiler = Compiler::<LlvmModule>::from_discrete_str(text).unwrap();
        let (n_states, n_inputs, n_outputs, _n_data, n_stop, has_mass) = compiler.get_dims();
        assert_eq!(n_states, 1);
        assert_eq!(n_inputs, 0);
        assert_eq!(n_outputs, 1);
        assert_eq!(n_stop, 0);
        assert!(!has_mass);
        let mut u0 = vec![0.];
        let mut res = vec![0.];
        let mut data = compiler.get_new_data();
        compiler.set_u0(u0.as_mut_slice(), data.as_mut_slice());
        assert_relative_eq!(u0.as_slice(), vec![1.].as_slice());
        compiler.rhs(0., u0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        assert_relative_eq!(res.as_slice(), vec![-1.].as_slice());
    }

    #[test]
    fn test_from_discrete_str_cranelift() {
        let text = "
        u { y = 1 }
        F { -y }
        out { y }
        ";
        let compiler = Compiler::<CraneliftModule>::from_discrete_str(text).unwrap();
        let (n_states, n_inputs, n_outputs, _n_data, n_stop, has_mass) = compiler.get_dims();
        assert_eq!(n_states, 1);
        assert_eq!(n_inputs, 0);
        assert_eq!(n_outputs, 1);
        assert_eq!(n_stop, 0);
        assert!(!has_mass);

        let mut u0 = vec![0.];
        let mut res = vec![0.];
        let mut data = compiler.get_new_data();
        compiler.set_u0(u0.as_mut_slice(), data.as_mut_slice());
        assert_relative_eq!(u0.as_slice(), vec![1.].as_slice());
        compiler.rhs(0., u0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        assert_relative_eq!(res.as_slice(), vec![-1.].as_slice());
    }

    #[test]
    fn test_stop_cranelift() {
        test_stop::<CraneliftModule>();
    }

    #[cfg(feature = "llvm")]
    #[test]
    fn test_stop_llvm() {
        test_stop::<crate::LlvmModule>();
    }

    fn test_stop<T: CodegenModule>() {
        let full_text = "
        u_i {
            y = 1,
        }
        dudt_i {
            dydt = 0,
        }
        M_i {
            dydt,
        }
        F_i {
            y * (1 - y),
        }
        stop_i {
            y - 0.5,
        }
        out {
            y,
        }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = DiscreteModel::build("$name", &model).unwrap();
        let compiler = Compiler::<T>::from_discrete_model(&discrete_model).unwrap();
        let mut u0 = vec![1.];
        let mut res = vec![0.];
        let mut stop = vec![0.];
        let mut data = compiler.get_new_data();
        compiler.set_u0(u0.as_mut_slice(), data.as_mut_slice());
        compiler.rhs(0., u0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        compiler.calc_stop(0., u0.as_slice(), data.as_mut_slice(), stop.as_mut_slice());
        assert_relative_eq!(stop[0], 0.5);
        assert_eq!(stop.len(), 1);
    }

    #[test]
    fn test_vector_add_scalar_cranelift() {
        let n = 1;
        let u = vec![1.0; n];
        let full_text = format!(
            "
            u_i {{
                {} 
            }}
            F_i {{
                u_i + 1.0,
            }}
            out_i {{
                u_i 
            }}
            ",
            (0..n)
                .map(|i| format!("x{} = {},", i, u[i]))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let model = parse_ds_string(&full_text).unwrap();
        let name = "$name";
        let discrete_model = DiscreteModel::build(name, &model).unwrap();
        env_logger::builder().is_test(true).try_init().unwrap();
        let _compiler = Compiler::<CraneliftModule>::from_discrete_model(&discrete_model).unwrap();
    }

    fn tensor_test_common<T: CodegenModule>(text: &str, tensor_name: &str) -> Vec<Vec<f64>> {
        let full_text = format!(
            "
            {}
        ",
            text
        );
        let model = parse_ds_string(full_text.as_str()).unwrap();
        let discrete_model = match DiscreteModel::build("$name", &model) {
            Ok(model) => model,
            Err(e) => {
                panic!("{}", e.as_error_message(full_text.as_str()));
            }
        };
        let compiler = Compiler::<T>::from_discrete_model(&discrete_model).unwrap();
        let mut u0 = vec![1.];
        let mut res = vec![0.];
        let mut data = compiler.get_new_data();
        let mut grad_data = Vec::new();
        let (_n_states, n_inputs, _n_outputs, _n_data, _n_stop, _has_mass) = compiler.get_dims();
        for _ in 0..n_inputs {
            grad_data.push(compiler.get_new_data());
        }
        let mut results = Vec::new();
        let inputs = vec![1.; n_inputs];
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice());
        compiler.set_u0(u0.as_mut_slice(), data.as_mut_slice());
        compiler.rhs(0., u0.as_slice(), data.as_mut_slice(), res.as_mut_slice());
        compiler.calc_out(0., u0.as_slice(), data.as_mut_slice());
        results.push(
            compiler
                .get_tensor_data(tensor_name, data.as_slice())
                .unwrap()
                .to_vec(),
        );
        for i in 0..n_inputs {
            let mut dinputs = vec![0.; n_inputs];
            dinputs[i] = 1.0;
            let mut ddata = compiler.get_new_data();
            let mut du0 = vec![0.];
            let mut dres = vec![0.];
            compiler.set_inputs_grad(
                inputs.as_slice(),
                dinputs.as_slice(),
                grad_data[i].as_mut_slice(),
                ddata.as_mut_slice(),
            );
            compiler.set_u0_grad(
                u0.as_mut_slice(),
                du0.as_mut_slice(),
                grad_data[i].as_mut_slice(),
                ddata.as_mut_slice(),
            );
            compiler.rhs_grad(
                0.,
                u0.as_slice(),
                du0.as_slice(),
                grad_data[i].as_mut_slice(),
                ddata.as_mut_slice(),
                res.as_mut_slice(),
                dres.as_mut_slice(),
            );
            compiler.calc_out_grad(
                0.,
                u0.as_slice(),
                du0.as_slice(),
                grad_data[i].as_mut_slice(),
                ddata.as_mut_slice(),
            );
            results.push(
                compiler
                    .get_tensor_data(tensor_name, ddata.as_slice())
                    .unwrap()
                    .to_vec(),
            );
        }
        results
    }

    macro_rules! tensor_test {
        ($($name:ident: $text:literal expect $tensor_name:literal $expected_value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let full_text = format!("
                    {}
                    u_i {{
                        y = 1,
                    }}
                    F_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", $text);

                #[cfg(feature = "llvm")]
                {
                    use crate::execution::llvm::codegen::LlvmModule;
                    let results = tensor_test_common::<LlvmModule>(full_text.as_str(), $tensor_name);
                    assert_relative_eq!(results[0].as_slice(), $expected_value.as_slice());
                }

                let results = tensor_test_common::<CraneliftModule>(full_text.as_str(), $tensor_name);
                assert_relative_eq!(results[0].as_slice(), $expected_value.as_slice());
            }
        )*
        }
    }

    tensor_test! {
        heaviside_function0: "r { heaviside(-0.1) }" expect "r" vec![0.0],
        heaviside_function1: "r { heaviside(0.0) }" expect "r" vec![1.0],
        exp_function: "r { exp(2) }" expect "r" vec![f64::exp(2.0)],
        pow_function: "r { pow(4.3245, 0.5) }" expect "r" vec![f64::powf(4.3245, 0.5)],
        arcsinh_function: "r { arcsinh(0.5) }" expect "r" vec![f64::asinh(0.5)],
        arccosh_function: "r { arccosh(2) }" expect "r" vec![f64::acosh(2.0)],
        tanh_function: "r { tanh(0.5) }" expect "r" vec![f64::tanh(0.5)],
        sinh_function: "r { sinh(0.5) }" expect "r" vec![f64::sinh(0.5)],
        cosh_function: "r { cosh(0.5) }" expect "r" vec![f64::cosh(0.5)],
        exp_function_time: "r { exp(t) }" expect "r" vec![f64::exp(0.0)],
        min_function: "r { min(2, 3) }" expect "r" vec![2.0],
        max_function: "r { max(2, 3) }" expect "r" vec![3.0],
        sigmoid_function: "r { sigmoid(0.1) }" expect "r" vec![1.0 / (1.0 + f64::exp(-0.1))],
        scalar: "r {2}" expect "r" vec![2.0,],
        constant: "r_i {2, 3}" expect "r" vec![2., 3.],
        expression: "r_i {2 + 3, 3 * 2, arcsinh(1.2 + 1.0 / max(1.2, 1.0) * 2.0 + tanh(2.0))}" expect "r" vec![5., 6., f64::asinh(1.2 + 1.0 / f64::max(1.2, 1.0) * 2.0 + f64::tanh(2.0))],
        pybamm_expression: "
        constant0_i { (0:19): 0.0, (19:20): 0.0006810238128045524,}
        constant1_i { (0:19): 0.0, (19:20): -0.0011634665332403958,}
        constant2_ij { (0,18): -25608.96286546366, (0,19): 76826.88859639116,}
        constant3_ij {(0,18): -0.4999999999999983, (0,19): 1.4999999999999984,}
        constant4_ij {(0,18): -0.4999999999999983, (0,19): 1.4999999999999982,}
        constant7_ij { (0,18): -12491.630996921805, (0,19): 37474.892990765504,}
        xaveragednegativeparticleconcentrationmolm3_i { 0.245049, 0.244694, 0.243985, 0.242921, 0.241503, 0.239730, 0.237603, 0.235121, 0.232284, 0.229093, 0.225547, 0.221647, 0.217392, 0.212783, 0.207819, 0.202500, 0.196827, 0.190799, 0.184417, 0.177680, }
        xaveragedpositiveparticleconcentrationmolm3_i { 0.939986, 0.940066, 0.940228, 0.940471, 0.940795, 0.941200, 0.941685, 0.942252, 0.942899, 0.943628, 0.944437, 0.945328, 0.946299, 0.947351, 0.948485, 0.949699, 0.950994, 0.952370, 0.953827, 0.955365, }
        varying2_i {(constant2_ij * xaveragedpositiveparticleconcentrationmolm3_j),}
        varying3_i {(constant4_ij * xaveragedpositiveparticleconcentrationmolm3_j),}
        varying4_i {(constant7_ij * xaveragednegativeparticleconcentrationmolm3_j),}
        varying5_i {(constant3_ij * xaveragednegativeparticleconcentrationmolm3_j),}
        r_i {(((0.05138515824298745 * arcsinh((-0.7999999999999998 / ((1.8973665961010275e-05 * pow(max(min(varying2_i, 51217.92521874824), 0.000512179257309275), 0.5)) * pow((51217.9257309275 - max(min(varying2_i, 51217.92521874824), 0.000512179257309275)), 0.5))))) + (((((((2.16216 + (0.07645 * tanh((30.834 - (57.858397200000006 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (2.1581 * tanh((52.294 - (53.412228 * max(min(varying3_i, 0.9999999999), 1e-10)))))) - (0.14169 * tanh((11.0923 - (21.0852666 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (0.2051 * tanh((1.4684 - (5.829105600000001 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (0.2531 * tanh((4.291641337386018 - (8.069908814589667 * max(min(varying3_i, 0.9999999999), 1e-10)))))) - (0.02167 * tanh((-87.5 + (177.0 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (1e-06 * ((1.0 / max(min(varying3_i, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + max(min(varying3_i, 0.9999999999), 1e-10))))))) - ((0.05138515824298745 * arcsinh((0.6666666666666666 / ((0.0006324555320336759 * pow(max(min(varying4_i, 24983.261744011077), 0.000249832619938437), 0.5)) * pow((24983.2619938437 - max(min(varying4_i, 24983.261744011077), 0.000249832619938437)), 0.5))))) + ((((((((((0.194 + (1.5 * exp((-120.0 * max(min(varying5_i, 0.9999999999), 1e-10))))) + (0.0351 * tanh((-3.44578313253012 + (12.048192771084336 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.0045 * tanh((-7.1344537815126055 + (8.403361344537815 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.035 * tanh((-18.466 + (20.0 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.0147 * tanh((-14.705882352941176 + (29.41176470588235 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.102 * tanh((-1.3661971830985917 + (7.042253521126761 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.022 * tanh((-54.8780487804878 + (60.975609756097555 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.011 * tanh((-5.486725663716814 + (44.24778761061947 * max(min(varying5_i, 0.9999999999), 1e-10)))))) + (0.0155 * tanh((-3.6206896551724133 + (34.48275862068965 * max(min(varying5_i, 0.9999999999), 1e-10)))))) + (1e-06 * ((1.0 / max(min(varying5_i, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + max(min(varying5_i, 0.9999999999), 1e-10)))))))),}
        " expect "r" vec![3.191533267340602],
        pybamm_subexpression: "
            constant2_ij { (0,18): -25608.96286546366, (0,19): 76826.88859639116,}
            st_i { (0:20): xaveragednegativeparticleconcentrationmolm3 = 0.8000000000000016, (20:40): xaveragedpositiveparticleconcentrationmolm3 = 0.6000000000000001, }
            varying2_i {(constant2_ij * xaveragedpositiveparticleconcentrationmolm3_j),}
        " expect "varying2" vec![-25608.96286546366 * 0.6000000000000001 + 76826.88859639116 * 0.6000000000000001],
        pybamm_subexpression2: "
            constant4_ij {(0,18): -0.4999999999999983, (0,19): 1.4999999999999982,}
            st_i { (0:20): xaveragednegativeparticleconcentrationmolm3 = 0.8000000000000016, (20:40): xaveragedpositiveparticleconcentrationmolm3 = 0.6000000000000001, }
            varying3_i {(constant4_ij * xaveragedpositiveparticleconcentrationmolm3_j),}
        " expect "varying3" vec![-0.4999999999999983 * 0.6000000000000001 + 1.4999999999999982 * 0.6000000000000001],
        pybamm_subexpression3: "
            constant7_ij { (0,18): -12491.630996921805, (0,19): 37474.892990765504,}
            st_i { (0:20): xaveragednegativeparticleconcentrationmolm3 = 0.8000000000000016, (20:40): xaveragedpositiveparticleconcentrationmolm3 = 0.6000000000000001, }
            varying4_i {(constant7_ij * xaveragednegativeparticleconcentrationmolm3_j),}
        " expect "varying4" vec![-12491.630996921805 * 0.8000000000000016 + 37474.892990765504 * 0.8000000000000016],
        pybamm_subexpression4: "
            varying2_i {30730.7554386,}
            varying3_i {0.6,}
            varying4_i {19986.6095951,}
            varying5_i {0.8,}
            r_i {(((0.05138515824298745 * arcsinh((-0.7999999999999998 / ((1.8973665961010275e-05 * pow(max(min(varying2_i, 51217.92521874824), 0.000512179257309275), 0.5)) * pow((51217.9257309275 - max(min(varying2_i, 51217.92521874824), 0.000512179257309275)), 0.5))))) + (((((((2.16216 + (0.07645 * tanh((30.834 - (57.858397200000006 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (2.1581 * tanh((52.294 - (53.412228 * max(min(varying3_i, 0.9999999999), 1e-10)))))) - (0.14169 * tanh((11.0923 - (21.0852666 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (0.2051 * tanh((1.4684 - (5.829105600000001 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (0.2531 * tanh((4.291641337386018 - (8.069908814589667 * max(min(varying3_i, 0.9999999999), 1e-10)))))) - (0.02167 * tanh((-87.5 + (177.0 * max(min(varying3_i, 0.9999999999), 1e-10)))))) + (1e-06 * ((1.0 / max(min(varying3_i, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + max(min(varying3_i, 0.9999999999), 1e-10))))))) - ((0.05138515824298745 * arcsinh((0.6666666666666666 / ((0.0006324555320336759 * pow(max(min(varying4_i, 24983.261744011077), 0.000249832619938437), 0.5)) * pow((24983.2619938437 - max(min(varying4_i, 24983.261744011077), 0.000249832619938437)), 0.5))))) + ((((((((((0.194 + (1.5 * exp((-120.0 * max(min(varying5_i, 0.9999999999), 1e-10))))) + (0.0351 * tanh((-3.44578313253012 + (12.048192771084336 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.0045 * tanh((-7.1344537815126055 + (8.403361344537815 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.035 * tanh((-18.466 + (20.0 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.0147 * tanh((-14.705882352941176 + (29.41176470588235 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.102 * tanh((-1.3661971830985917 + (7.042253521126761 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.022 * tanh((-54.8780487804878 + (60.975609756097555 * max(min(varying5_i, 0.9999999999), 1e-10)))))) - (0.011 * tanh((-5.486725663716814 + (44.24778761061947 * max(min(varying5_i, 0.9999999999), 1e-10)))))) + (0.0155 * tanh((-3.6206896551724133 + (34.48275862068965 * max(min(varying5_i, 0.9999999999), 1e-10)))))) + (1e-06 * ((1.0 / max(min(varying5_i, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + max(min(varying5_i, 0.9999999999), 1e-10)))))))),}
        " expect "r" vec![(((0.05138515824298745 * f64::asinh(-0.7999999999999998 / ((1.897_366_596_101_027_5e-5 * f64::powf(f64::max(f64::min(30730.7554386, 51217.92521874824), 0.000512179257309275), 0.5)) * f64::powf(51217.9257309275 - f64::max(f64::min(30730.7554386, 51217.92521874824), 0.000512179257309275), 0.5)))) + (((((((2.16216 + (0.07645 * f64::tanh(30.834 - (57.858397200000006 * f64::max(f64::min(0.6, 0.9999999999), 1e-10))))) + (2.1581 * f64::tanh(52.294 - (53.412228 * f64::max(f64::min(0.6, 0.9999999999), 1e-10))))) - (0.14169 * f64::tanh(11.0923 - (21.0852666 * f64::max(f64::min(0.6, 0.9999999999), 1e-10))))) + (0.2051 * f64::tanh(1.4684 - (5.829105600000001 * f64::max(f64::min(0.6, 0.9999999999), 1e-10))))) + (0.2531 * f64::tanh(4.291641337386018 - (8.069908814589667 * f64::max(f64::min(0.6, 0.9999999999), 1e-10))))) - (0.02167 * f64::tanh(-87.5 + (177.0 * f64::max(f64::min(0.6, 0.9999999999), 1e-10))))) + (1e-06 * ((1.0 / f64::max(f64::min(0.6, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + f64::max(f64::min(0.6, 0.9999999999), 1e-10))))))) - ((0.05138515824298745 * f64::asinh(0.6666666666666666 / ((0.0006324555320336759 * f64::powf(f64::max(f64::min(19986.6095951, 24983.261744011077), 0.000249832619938437), 0.5)) * f64::powf(24983.2619938437 - f64::max(f64::min(19986.6095951, 24983.261744011077), 0.000249832619938437), 0.5)))) + ((((((((((0.194 + (1.5 * f64::exp(-120.0 * f64::max(f64::min(0.8, 0.9999999999), 1e-10)))) + (0.0351 * f64::tanh(-3.44578313253012 + (12.048192771084336 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) - (0.0045 * f64::tanh(-7.1344537815126055 + (8.403361344537815 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) - (0.035 * f64::tanh(-18.466 + (20.0 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) - (0.0147 * f64::tanh(-14.705882352941176 + (29.41176470588235 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) - (0.102 * f64::tanh(-1.3661971830985917 + (7.042253521126761 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) - (0.022 * f64::tanh(-54.8780487804878 + (60.975609756097555 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) - (0.011 * f64::tanh(-5.486725663716814 + (44.24778761061947 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) + (0.0155 * f64::tanh(-3.6206896551724133 + (34.48275862068965 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))))) + (1e-06 * ((1.0 / f64::max(f64::min(0.8, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + f64::max(f64::min(0.8, 0.9999999999), 1e-10))))))))],
        pybamm_subexpression5: "r_i { (1.0 / max(min(0.6, 0.9999999999), 1e-10)),}" expect "r" vec![1.0 / f64::max(f64::min(0.6, 0.9999999999), 1e-10)],
        pybamm_subexpression6: "r_i { arcsinh(1.8973665961010275e-05), }" expect "r" vec![f64::asinh(1.897_366_596_101_027_5e-5)],
        pybamm_subexpression7: "r_i { (1.5 * exp(-120.0 * max(min(0.8, 0.9999999999), 1e-10))), }" expect "r" vec![1.5 * f64::exp(-120.0 * f64::max(f64::min(0.8, 0.9999999999), 1e-10))],
        pybamm_subexpression8: "r_i { (0.07645 * tanh(30.834 - (57.858397200000006 * max(min(0.6, 0.9999999999), 1e-10)))), }" expect "r" vec![0.07645 * f64::tanh(30.834 - (57.858397200000006 * f64::max(f64::min(0.6, 0.9999999999), 1e-10)))],
        pybamm_subexpression9: "r_i { (1e-06 * ((1.0 / max(min(0.8, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + max(min(0.8, 0.9999999999), 1e-10))))), }" expect "r" vec![1e-06 * ((1.0 / f64::max(f64::min(0.8, 0.9999999999), 1e-10)) + (1.0 / (-1.0 + f64::max(f64::min(0.8, 0.9999999999), 1e-10))))],
        pybamm_subexpression10: "r_i { (1.0 / (-1.0 + max(min(0.8, 0.9999999999), 1e-10))), }" expect "r" vec![1.0 / (-1.0 + f64::max(f64::min(0.8, 0.9999999999), 1e-10))],
        unary_negate_in_expr: "r_i { 1.0 / (-1.0 + 1.1) }" expect "r" vec![1.0 / (-1.0 + 1.1)],
        derived: "r_i {2, 3} k_i { 2 * r_i }" expect "k" vec![4., 6.],
        concatenate: "r_i {2, 3} k_i { r_i, 2 * r_i }" expect "k" vec![2., 3., 4., 6.],
        ones_matrix_dense: "I_ij { (0:2, 0:2): 1 }" expect "I" vec![1., 1., 1., 1.],
        dense_matrix: "A_ij { (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 }" expect "A" vec![1., 2., 3., 4.],
        dense_vector: "x_i { (0:4): 1, (4:5): 2 }" expect "x" vec![1., 1., 1., 1., 2.],
        identity_matrix_diagonal: "I_ij { (0..2, 0..2): 1 }" expect "I" vec![1., 1.],
        concatenate_diagonal: "A_ij { (0..2, 0..2): 1 } B_ij { (0:2, 0:2): A_ij, (2:4, 2:4): A_ij }" expect "B" vec![1., 1., 1., 1.],
        identity_matrix_sparse: "I_ij { (0, 0): 1, (1, 1): 2 }" expect "I" vec![1., 2.],
        concatenate_sparse: "A_ij { (0, 0): 1, (1, 1): 2 } B_ij { (0:2, 0:2): A_ij, (2:4, 2:4): A_ij }" expect "B" vec![1., 2., 1., 2.],
        sparse_rearrange: "A_ij { (0, 0): 1, (1, 1): 2, (0, 1): 3 }" expect "A" vec![1., 3., 2.],
        sparse_rearrange2: "A_ij { (0, 1): 1, (1, 1): 2, (1, 0): 3, (2, 2): 4, (2, 1): 5 }" expect "A" vec![1., 3., 2., 5., 4.],
        sparse_expression: "A_ij { (0, 0): 1, (0, 1): 2, (1, 1): 3 } B_ij { 2 * A_ij }" expect "B" vec![2., 4., 6.],
        sparse_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 0): 2, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" vec![1., 8.],
        sparse_rearrange_matrix_vect_multiply: "A_ij { (0, 1): 1, (1, 1): 2, (1, 0): 3, (2, 2): 4, (2, 1): 5 } x_i { 1, 2, 3 } b_i { A_ij * x_j }" expect "b" vec![2., 7., 22.],
        diag_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" vec![1., 6.],
        dense_matrix_vect_multiply: "A_ij {  (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" vec![5., 11.],
        sparse_matrix_vect_multiply_zero_row: "A_ij { (0, 1): 2 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" vec![4.],
    }

    macro_rules! tensor_grad_test {
        ($($name:ident: $text:literal expect $tensor_name:literal $expected_value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let full_text = format!("
                    in = [p]
                    p {{
                        1,
                    }}
                    u_i {{
                        y = p,
                    }}
                    dudt_i {{
                        dydt = p,
                    }}
                    {}
                    M_i {{
                        dydt,
                    }}
                    F_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", $text);

                #[cfg(feature = "llvm")]
                {
                    use crate::execution::llvm::codegen::LlvmModule;
                    let results = tensor_test_common::<LlvmModule>(full_text.as_str(), $tensor_name);
                    assert_relative_eq!(results[1].as_slice(), $expected_value.as_slice());
                }

                let results = tensor_test_common::<CraneliftModule>(full_text.as_str(), $tensor_name);
                assert_relative_eq!(results[1].as_slice(), $expected_value.as_slice());
            }
        )*
        }
    }

    tensor_grad_test! {
        const_grad: "r { 3 }" expect "r" vec![0.],
        const_vec_grad: "r_i { 3, 4 }" expect "r" vec![0., 0.],
        input_grad: "r { 2 * p * p }" expect "r" vec![4.],
        input_vec_grad: "r_i { 2 * p * p, 3 * p }" expect "r" vec![4., 3.],
        state_grad: "r { 2 * y }" expect "r" vec![2.],
        input_and_state_grad: "r { 2 * y * p }" expect "r" vec![4.],
    }

    #[test]
    fn test_repeated_grad_cranelift() {
        test_repeated_grad_common::<CraneliftModule>();
    }

    #[cfg(feature = "llvm")]
    #[test]
    fn test_repeated_grad_llvm() {
        test_repeated_grad_common::<crate::LlvmModule>();
    }

    fn test_repeated_grad_common<T: CodegenModule>() {
        let full_text = "
            in = [p]
            p {
                1,
            }
            u_i {
                y = p,
            }
            dudt_i {
                dydt = 1,
            }
            r {
                2 * y * p,
            }
            M_i {
                dydt,
            }
            F_i {
                r,
            }
            out_i {
                y,
            }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = match DiscreteModel::build("test_repeated_grad", &model) {
            Ok(model) => model,
            Err(e) => {
                panic!("{}", e.as_error_message(full_text));
            }
        };
        let compiler = Compiler::<T>::from_discrete_model(&discrete_model).unwrap();
        let mut u0 = vec![1.];
        let mut du0 = vec![1.];
        let mut res = vec![0.];
        let mut dres = vec![0.];
        let mut data = compiler.get_new_data();
        let mut ddata = compiler.get_new_data();
        let (_n_states, n_inputs, _n_outputs, _n_data, _n_stop, _has_mass) = compiler.get_dims();

        for _i in 0..3 {
            let inputs = vec![2.; n_inputs];
            let dinputs = vec![1.; n_inputs];
            compiler.set_inputs_grad(
                inputs.as_slice(),
                dinputs.as_slice(),
                data.as_mut_slice(),
                ddata.as_mut_slice(),
            );
            compiler.set_u0_grad(
                u0.as_mut_slice(),
                du0.as_mut_slice(),
                data.as_mut_slice(),
                ddata.as_mut_slice(),
            );
            compiler.rhs_grad(
                0.,
                u0.as_slice(),
                du0.as_slice(),
                data.as_mut_slice(),
                ddata.as_mut_slice(),
                res.as_mut_slice(),
                dres.as_mut_slice(),
            );
            assert_relative_eq!(dres.as_slice(), vec![8.].as_slice());
        }
    }

    #[test]
    fn test_additional_functions() {
        let full_text = "
            in = [k]
            k {
                1,
            }
            u_i {
                y = 1,
                x = 2,
            }
            dudt_i {
                dydt = 0,
                0,
            }
            M_i {
                dydt,
                0,
            }
            F_i {
                y - 1,
                x - 2,
            }
            out_i {
                y,
                x,
                2*x,
            }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = DiscreteModel::build("$name", &model).unwrap();
        let compiler = Compiler::<CraneliftModule>::from_discrete_model(&discrete_model).unwrap();
        let (n_states, n_inputs, n_outputs, n_data, _n_stop, _has_mass) = compiler.get_dims();
        assert_eq!(n_states, 2);
        assert_eq!(n_inputs, 1);
        assert_eq!(n_outputs, 3);
        assert_eq!(n_data, compiler.data_len());

        let mut data = compiler.get_new_data();
        let inputs = vec![1.1];
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice());

        let inputs = compiler.get_tensor_data("k", data.as_slice()).unwrap();
        assert_relative_eq!(inputs, vec![1.1].as_slice());

        let mut id = vec![0.0, 0.0];
        compiler.set_id(id.as_mut_slice());
        assert_eq!(id, vec![1.0, 0.0]);

        let mut u = vec![0., 0.];
        compiler.set_u0(u.as_mut_slice(), data.as_mut_slice());
        assert_relative_eq!(u.as_slice(), vec![1., 2.].as_slice());

        let mut rr = vec![1., 1.];
        compiler.rhs(0., u.as_slice(), data.as_mut_slice(), rr.as_mut_slice());
        assert_relative_eq!(rr.as_slice(), vec![0., 0.].as_slice());

        let up = vec![2., 3.];
        rr = vec![1., 1.];
        compiler.mass(0., up.as_slice(), data.as_mut_slice(), rr.as_mut_slice());
        assert_relative_eq!(rr.as_slice(), vec![2., 0.].as_slice());

        compiler.calc_out(0., u.as_slice(), data.as_mut_slice());
        let out = compiler.get_out(data.as_slice());
        assert_relative_eq!(out, vec![1., 2., 4.].as_slice());
    }

    #[test]
    fn test_inputs() {
        let full_text = "
            in = [c, a, b]
            a { 1 } b { 2 } c { 3 }
            u { y = 0 }
            F { y }
            out { y }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = DiscreteModel::build("test_inputs", &model).unwrap();

        let compiler = Compiler::<CraneliftModule>::from_discrete_model(&discrete_model).unwrap();
        let mut data = compiler.get_new_data();
        let inputs = vec![1.0, 2.0, 3.0];
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice());

        for (name, expected_value) in [("a", vec![2.0]), ("b", vec![3.0]), ("c", vec![1.0])] {
            let inputs = compiler.get_tensor_data(name, data.as_slice()).unwrap();
            assert_relative_eq!(inputs, expected_value.as_slice());
        }

        #[cfg(feature = "llvm")]
        {
            let compiler =
                Compiler::<crate::LlvmModule>::from_discrete_model(&discrete_model).unwrap();
            let mut data = compiler.get_new_data();
            let inputs = vec![1.0, 2.0, 3.0];
            compiler.set_inputs(inputs.as_slice(), data.as_mut_slice());

            for (name, expected_value) in [("a", vec![2.0]), ("b", vec![3.0]), ("c", vec![1.0])] {
                let inputs = compiler.get_tensor_data(name, data.as_slice()).unwrap();
                assert_relative_eq!(inputs, expected_value.as_slice());
            }
        }
    }
}
