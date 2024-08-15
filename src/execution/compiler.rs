use crate::{discretise::DiscreteModel, execution::interface::{CalcOutFunc, GetDimsFunc, GetOutFunc, MassFunc, RhsFunc, SetIdFunc, SetInputsFunc, StopFunc, U0Func}, parser::parse_ds_string};

use super::{interface::{CalcOutGradientFunc, RhsGradientFunc, SetInputsGradientFunc, U0GradientFunc}, module::CodegenModule};
use anyhow::Result;
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
        let model = parse_ds_string(code).unwrap();
        let model = DiscreteModel::build(name.as_str(), &model)
            .unwrap_or_else(|e| panic!("{}", e.as_error_message(code)));
        Self::from_discrete_model(&model)
    }

    pub fn from_discrete_model(model: &DiscreteModel) -> Result<Self> {
        let number_of_states = *model.state().shape().first().unwrap_or(&1);
        let input_names = model
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect::<Vec<_>>();
        let mut module = M::new(Triple::host(), model);
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

        let set_u0_grad = module.compile_set_u0_grad(&set_u0)?;
        let rhs_grad = module.compile_rhs_grad(&rhs)?;
        let calc_out_grad = module.compile_calc_out_grad(&calc_out)?;
        let set_inputs_grad = module.compile_set_inputs_grad(&set_inputs)?;

        module.post_autodiff_optimisation()?;

        let set_u0 = unsafe { std::mem::transmute::<*const u8, U0Func>(module.jit(set_u0)?) };
        let rhs = unsafe { std::mem::transmute::<*const u8, RhsFunc>(module.jit(rhs)?) };
        let mass = unsafe { std::mem::transmute::<*const u8, MassFunc>(module.jit(mass)?) };
        let calc_out = unsafe { std::mem::transmute::<*const u8, CalcOutFunc>(module.jit(calc_out)?) };
        let calc_stop = unsafe { std::mem::transmute::<*const u8, StopFunc>(module.jit(calc_stop)?) };
        let set_id = unsafe { std::mem::transmute::<*const u8, SetIdFunc>(module.jit(set_id)?) };
        let get_dims = unsafe { std::mem::transmute::<*const u8, GetDimsFunc>(module.jit(get_dims)?) };
        let set_inputs = unsafe { std::mem::transmute::<*const u8, SetInputsFunc>(module.jit(set_inputs)?) };
        let get_out = unsafe { std::mem::transmute::<*const u8, GetOutFunc>(module.jit(get_output)?) };

        let set_u0_grad = unsafe { std::mem::transmute::<*const u8, U0GradientFunc>(module.jit(set_u0_grad)?) };
        let rhs_grad = unsafe { std::mem::transmute::<*const u8, RhsGradientFunc>(module.jit(rhs_grad)?) };
        let calc_out_grad = unsafe { std::mem::transmute::<*const u8, CalcOutGradientFunc>(module.jit(calc_out_grad)?) };
        let set_inputs_grad = unsafe { std::mem::transmute::<*const u8, SetInputsGradientFunc>(module.jit(set_inputs_grad)?) };


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
            panic!("Expected {} states, got {}", self.number_of_states, yy.len());
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
            panic!("Expected {} states, got {}", self.number_of_states, yy.len());
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
        unsafe { (self.jit_grad_functions.set_u0_grad)(data.as_mut_ptr(), ddata.as_mut_ptr(), yy.as_mut_ptr(), dyy.as_mut_ptr()) };
    }

    pub fn calc_stop(&self, t: f64, yy: &[f64], data: &mut [f64], stop: &mut [f64]) {
        let (n_states, _, _, n_data, n_stop) = self.get_dims();
        if yy.len() != n_states {
            panic!("Expected {} states, got {}", n_states, yy.len());
        }
        if data.len() != n_data {
            panic!("Expected {} data, got {}", n_data, data.len());
        }
        if stop.len() != n_stop {
            panic!("Expected {} stop, got {}", n_stop, stop.len());
        }
        unsafe { (self.jit_functions.calc_stop)(t, yy.as_ptr(), data.as_mut_ptr(), stop.as_mut_ptr()) };
    }

    pub fn rhs(&self, t: f64, yy: &[f64], data: &mut [f64], rr: &mut [f64]) {
        if yy.len() != self.number_of_states {
            panic!("Expected {} states, got {}", self.number_of_states, yy.len());
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
            panic!("Expected {} states, got {}", self.number_of_states, yp.len());
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
            panic!("Expected {} states, got {}", self.number_of_states, yy.len());
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
        unsafe { (self.jit_grad_functions.rhs_grad)(t, yy.as_ptr(), dyy.as_ptr(), data.as_mut_ptr(), ddata.as_mut_ptr(), rr.as_mut_ptr(), drr.as_mut_ptr()) };
    }

    pub fn calc_out(&self, t: f64, yy: &[f64], data: &mut [f64]) {
        if yy.len() != self.number_of_states {
            panic!("Expected {} states, got {}", self.number_of_states, yy.len());
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
            panic!("Expected {} states, got {}", self.number_of_states, yy.len());
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
        unsafe { (self.jit_grad_functions.calc_out_grad)(t, yy.as_ptr(), dyy.as_ptr(), data.as_mut_ptr(), ddata.as_mut_ptr()) };
    }

    /// Get various dimensions of the model
    ///
    /// # Returns
    ///
    /// A tuple of the form `(n_states, n_inputs, n_outputs, n_data, n_stop)`
    pub fn get_dims(&self) -> (usize, usize, usize, usize, usize) {
        let mut n_states = 0u32;
        let mut n_inputs = 0u32;
        let mut n_outputs = 0u32;
        let mut n_data = 0u32;
        let mut n_stop = 0u32;
        unsafe {(self.jit_functions.get_dims)(&mut n_states, &mut n_inputs, &mut n_outputs, &mut n_data, &mut n_stop)};
        (
            n_states as usize,
            n_inputs as usize,
            n_outputs as usize,
            n_data as usize,
            n_stop as usize,
        )
    }

    pub fn set_inputs(&self, inputs: &[f64], data: &mut [f64]) {
        let (_, n_inputs, _, _, _) = self.get_dims();
        if n_inputs != inputs.len() {
            panic!("Expected {} inputs, got {}", n_inputs, inputs.len());
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        unsafe {(self.jit_functions.set_inputs)(inputs.as_ptr(), data.as_mut_ptr())};
    }

    pub fn set_inputs_grad(
        &self,
        inputs: &[f64],
        dinputs: &[f64],
        data: &mut [f64],
        ddata: &mut [f64],
    ) {
        let (_, n_inputs, _, _, _) = self.get_dims();
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
        unsafe {(self.jit_grad_functions.set_inputs_grad)(inputs.as_ptr(), dinputs.as_ptr(), data.as_mut_ptr(), ddata.as_mut_ptr())};
    }

    pub fn get_out(&self, data: &[f64]) -> &[f64] {
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        let (_, _, n_outputs, _, _) = self.get_dims();
        let mut tensor_data_ptr: *mut f64 = std::ptr::null_mut();
        let mut tensor_data_len = 0u32;
        let tensor_data_ptr_ptr: *mut *mut f64 = &mut tensor_data_ptr;
        let tensor_data_len_ptr: *mut u32 = &mut tensor_data_len;
        unsafe {(self.jit_functions.get_out)(data.as_ptr(), tensor_data_ptr_ptr, tensor_data_len_ptr)};
        assert!(tensor_data_len as usize == n_outputs);
        unsafe { std::slice::from_raw_parts(tensor_data_ptr, tensor_data_len as usize) }
    }

    pub fn set_id(&self, id: &mut [f64]) {
        let (n_states, _, _, _, _) = self.get_dims();
        if n_states != id.len() {
            panic!("Expected {} states, got {}", n_states, id.len());
        }
        unsafe {(self.jit_functions.set_id)(id.as_mut_ptr())};
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
}