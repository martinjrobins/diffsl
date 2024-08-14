use std::env;

use crate::{discretise::DiscreteModel, execution::interface::{CalcOutFunc, GetDimsFunc, GetOutFunc, MassFunc, RhsFunc, SetIdFunc, SetInputsFunc, StopFunc, U0Func}, parser::parse_ds_string};

use super::{interface::{CalcOutGradientFunc, RhsGradientFunc, SetInputsGradientFunc, U0GradientFunc}, module::CodeGenModule};
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

pub struct Compiler<M: CodeGenModule> {
    module: M,
    jit_functions: JitFunctions,
    jit_grad_functions: JitGradFunctions,

    number_of_states: usize,
    number_of_parameters: usize,
    number_of_outputs: usize,
    has_mass: bool,
    output_base_filename: String,
}

impl<M: CodeGenModule> Compiler<M> {
    pub fn from_discrete_str(code: &str) -> Result<Self> {
        let uid = Id::<u32>::new();
        let name = format!("diffsl_{}", uid);
        let model = parse_ds_string(code).unwrap();
        let model = DiscreteModel::build(name.as_str(), &model)
            .unwrap_or_else(|e| panic!("{}", e.as_error_message(code)));
        let dir = env::temp_dir();
        let path = dir.join(name.clone());
        Self::from_discrete_model(&model, path.to_str().unwrap())
    }

    pub fn from_discrete_model(model: &DiscreteModel, out: &str) -> Result<Self> {
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

        let set_u0_grad = module.compile_set_u0_grad(set_u0)?;
        let rhs_grad = module.compile_rhs_grad(rhs)?;
        let calc_out_grad = module.compile_calc_out_grad(calc_out)?;
        let set_inputs_grad = module.compile_set_inputs_grad(set_inputs)?;

        module.post_autodiff_optimisation()?;

        let set_u0 = module.jit(set_u0)?;
        let rhs = module.jit(rhs)?;
        let mass = module.jit(mass)?;
        let calc_stop = module.jit(calc_stop)?;
        let calc_out = module.jit(calc_out)?;
        let set_id = module.jit(set_id)?;
        let get_dims = module.jit(get_dims)?;
        let set_inputs = module.jit(set_inputs)?;
        let get_out= module.jit(get_output)?;

        let set_u0_grad = module.jit(set_u0_grad)?;
        let rhs_grad = module.jit(rhs_grad)?;
        let calc_out_grad = module.jit(calc_out_grad)?;
        let set_inputs_grad = module.jit(set_inputs_grad)?;

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
            output_base_filename: out.to_string(),
        })
    }


    fn get_bitcode_filename(out: &str) -> String {
        format!("{}.bc", out)
    }

    fn get_object_filename(out: &str) -> String {
        format!("{}.o", out)
    }

    fn jit<'ctx, T>(name: &str, ee: &ExecutionEngine<'ctx>) -> Result<JitFunction<'ctx, T>>
    where
        T: UnsafeFunctionPointer,
    {
        let maybe_fn = unsafe { ee.get_function::<T>(name) };
        match maybe_fn {
            Ok(f) => Ok(f),
            Err(err) => Err(anyhow!("Error during jit for {}: {}", name, err)),
        }
    }

    pub fn get_tensor_data<'a>(&self, name: &str, data: &'a [f64]) -> Option<&'a [f64]> {
        let index = self.borrow_data_layout().get_data_index(name)?;
        let nnz = self.borrow_data_layout().get_data_length(name)?;
        Some(&data[index..index + nnz])
    }

    pub fn set_u0(&self, yy: &mut [f64], data: &mut [f64]) {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            panic!("Expected {} states, got {}", number_of_states, yy.len());
        }
        self.with_data(|compiler| {
            let yy_ptr = yy.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            unsafe {
                compiler.jit_functions.set_u0.call(data_ptr, yy_ptr);
            }
        });
    }

    pub fn set_u0_grad(
        &self,
        yy: &mut [f64],
        dyy: &mut [f64],
        data: &mut [f64],
        ddata: &mut [f64],
    ) {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            panic!("Expected {} states, got {}", number_of_states, yy.len());
        }
        if dyy.len() != number_of_states {
            panic!(
                "Expected {} states for dyy, got {}",
                number_of_states,
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
        self.with_data(|compiler| {
            let yy_ptr = yy.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            let dyy_ptr = dyy.as_mut_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            unsafe {
                compiler
                    .jit_grad_functions
                    .set_u0_grad
                    .call(data_ptr, ddata_ptr, yy_ptr, dyy_ptr);
            }
        });
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
        self.with_data(|compiler| {
            let yy_ptr = yy.as_ptr();
            let data_ptr = data.as_mut_ptr();
            let stop_ptr = stop.as_mut_ptr();
            unsafe {
                compiler
                    .jit_functions
                    .calc_stop
                    .call(t, yy_ptr, data_ptr, stop_ptr);
            }
        });
    }

    pub fn rhs(&self, t: f64, yy: &[f64], data: &mut [f64], rr: &mut [f64]) {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            panic!("Expected {} states, got {}", number_of_states, yy.len());
        }
        if rr.len() != number_of_states {
            panic!(
                "Expected {} residual states, got {}",
                number_of_states,
                rr.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        self.with_data(|compiler| {
            let yy_ptr = yy.as_ptr();
            let rr_ptr = rr.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            unsafe {
                compiler.jit_functions.rhs.call(t, yy_ptr, data_ptr, rr_ptr);
            }
        });
    }

    pub fn has_mass(&self) -> bool {
        *self.borrow_has_mass()
    }

    pub fn mass(&self, t: f64, yp: &[f64], data: &mut [f64], rr: &mut [f64]) {
        if !self.borrow_has_mass() {
            panic!("Model does not have a mass function");
        }
        let number_of_states = *self.borrow_number_of_states();
        if yp.len() != number_of_states {
            panic!("Expected {} states, got {}", number_of_states, yp.len());
        }
        if rr.len() != number_of_states {
            panic!(
                "Expected {} residual states, got {}",
                number_of_states,
                rr.len()
            );
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        self.with_data(|compiler| {
            let yp_ptr = yp.as_ptr();
            let rr_ptr = rr.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            unsafe {
                compiler
                    .jit_functions
                    .mass
                    .call(t, yp_ptr, data_ptr, rr_ptr);
            }
        });
    }

    pub fn data_len(&self) -> usize {
        self.with(|compiler| compiler.data_layout.data().len())
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
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            panic!("Expected {} states, got {}", number_of_states, yy.len());
        }
        if rr.len() != number_of_states {
            panic!(
                "Expected {} residual states, got {}",
                number_of_states,
                rr.len()
            );
        }
        if dyy.len() != number_of_states {
            panic!(
                "Expected {} states for dyy, got {}",
                number_of_states,
                dyy.len()
            );
        }
        if drr.len() != number_of_states {
            panic!(
                "Expected {} residual states for drr, got {}",
                number_of_states,
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
        self.with_data(|compiler| {
            let yy_ptr = yy.as_ptr();
            let rr_ptr = rr.as_mut_ptr();
            let dyy_ptr = dyy.as_ptr();
            let drr_ptr = drr.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            unsafe {
                compiler
                    .jit_grad_functions
                    .rhs_grad
                    .call(t, yy_ptr, dyy_ptr, data_ptr, ddata_ptr, rr_ptr, drr_ptr);
            }
        });
    }

    pub fn calc_out(&self, t: f64, yy: &[f64], data: &mut [f64]) {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != *self.borrow_number_of_states() {
            panic!("Expected {} states, got {}", number_of_states, yy.len());
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        self.with_data(|compiler| {
            let yy_ptr = yy.as_ptr();
            let data_ptr = data.as_mut_ptr();
            unsafe {
                compiler.jit_functions.calc_out.call(t, yy_ptr, data_ptr);
            }
        });
    }

    pub fn calc_out_grad(
        &self,
        t: f64,
        yy: &[f64],
        dyy: &[f64],
        data: &mut [f64],
        ddata: &mut [f64],
    ) {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != *self.borrow_number_of_states() {
            panic!("Expected {} states, got {}", number_of_states, yy.len());
        }
        if data.len() != self.data_len() {
            panic!("Expected {} data, got {}", self.data_len(), data.len());
        }
        if dyy.len() != *self.borrow_number_of_states() {
            panic!(
                "Expected {} states for dyy, got {}",
                number_of_states,
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
        self.with_data(|compiler| {
            let yy_ptr = yy.as_ptr();
            let data_ptr = data.as_mut_ptr();
            let dyy_ptr = dyy.as_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            unsafe {
                compiler
                    .jit_grad_functions
                    .calc_out_grad
                    .call(t, yy_ptr, dyy_ptr, data_ptr, ddata_ptr);
            }
        });
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
        self.with(|compiler| unsafe {
            compiler.data.jit_functions.get_dims.call(
                &mut n_states,
                &mut n_inputs,
                &mut n_outputs,
                &mut n_data,
                &mut n_stop,
            );
        });
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
        self.with_data(|compiler| {
            let data_ptr = data.as_mut_ptr();
            unsafe {
                compiler
                    .jit_functions
                    .set_inputs
                    .call(inputs.as_ptr(), data_ptr);
            }
        });
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
        self.with_data(|compiler| {
            let data_ptr = data.as_mut_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            let dinputs_ptr = dinputs.as_ptr();
            unsafe {
                compiler.jit_grad_functions.set_inputs_grad.call(
                    inputs.as_ptr(),
                    dinputs_ptr,
                    data_ptr,
                    ddata_ptr,
                );
            }
        });
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
        self.with(|compiler| {
            let data_ptr = data.as_ptr();
            unsafe {
                compiler.data.jit_functions.get_out.call(
                    data_ptr,
                    tensor_data_ptr_ptr,
                    tensor_data_len_ptr,
                );
            }
        });
        assert!(tensor_data_len as usize == n_outputs);
        unsafe { std::slice::from_raw_parts(tensor_data_ptr, tensor_data_len as usize) }
    }

    pub fn set_id(&self, id: &mut [f64]) {
        let (n_states, _, _, _, _) = self.get_dims();
        if n_states != id.len() {
            panic!("Expected {} states, got {}", n_states, id.len());
        }
        self.with_data(|compiler| {
            unsafe {
                compiler.jit_functions.set_id.call(id.as_mut_ptr());
            };
        });
    }

    fn get_native_machine() -> Result<TargetMachine> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| anyhow!("{}", e))?;
        let opt = OptimizationLevel::Default;
        let reloc = RelocMode::Default;
        let model = CodeModel::Default;
        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple).unwrap();
        let target_machine = target
            .create_target_machine(
                &target_triple,
                TargetMachine::get_host_cpu_name().to_str().unwrap(),
                TargetMachine::get_host_cpu_features().to_str().unwrap(),
                opt,
                reloc,
                model,
            )
            .unwrap();
        Ok(target_machine)
    }

    fn get_wasm_machine() -> Result<TargetMachine> {
        Target::initialize_webassembly(&InitializationConfig::default());
        let opt = OptimizationLevel::Default;
        let reloc = RelocMode::Default;
        let model = CodeModel::Default;
        let target_triple = TargetTriple::create("wasm32-unknown-emscripten");
        let target = Target::from_triple(&target_triple).unwrap();
        let target_machine = target
            .create_target_machine(&target_triple, "generic", "", opt, reloc, model)
            .unwrap();
        Ok(target_machine)
    }

    pub fn write_bitcode_to_path(&self, path: &Path) -> Result<()> {
        self.with_data(|data| {
            let result = data.codegen.module().write_bitcode_to_path(path);
            if result {
                Ok(())
            } else {
                Err(anyhow!("Error writing bitcode to path"))
            }
        })
    }

    pub fn write_object_file(&self, path: &Path) -> Result<()> {
        let target_machine = LlvmCompiler::get_native_machine()?;
        self.with_data(|data| {
            target_machine
                .write_to_file(data.codegen.module(), FileType::Object, path)
                .map_err(|e| anyhow::anyhow!("Error writing object file: {:?}", e))
        })
    }

    pub fn write_wasm_object_file(&self, path: &Path) -> Result<()> {
        let target_machine = LlvmCompiler::get_wasm_machine()?;
        self.with_data(|data| {
            target_machine
                .write_to_file(data.codegen.module(), FileType::Object, path)
                .map_err(|e| anyhow::anyhow!("Error writing object file: {:?}", e))
        })
    }

    pub fn number_of_states(&self) -> usize {
        *self.borrow_number_of_states()
    }
    pub fn number_of_parameters(&self) -> usize {
        *self.borrow_number_of_parameters()
    }

    pub fn number_of_outputs(&self) -> usize {
        *self.borrow_number_of_outputs()
    }
}