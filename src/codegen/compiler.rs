use std::path::Path;
use anyhow::anyhow;

use anyhow::Result;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::targets::TargetMachine;
use inkwell::{context::Context, OptimizationLevel, targets::{TargetTriple, InitializationConfig, Target, RelocMode, CodeModel, FileType}, execution_engine::{JitFunction, ExecutionEngine, UnsafeFunctionPointer}};
use ouroboros::self_referencing;
use crate::discretise::DiscreteModel;
use crate::utils::find_executable;
use crate::utils::find_library_path;
use crate::utils::find_runtime_path;
use std::process::Command;


use super::codegen::CalcOutGradientFunc;
use super::codegen::CompileGradientArgType;
use super::codegen::GetDimsFunc;
use super::codegen::GetOutFunc;
use super::codegen::ResidualGradientFunc;
use super::codegen::SetIdFunc;
use super::codegen::SetInputsFunc;
use super::codegen::SetInputsGradientFunc;
use super::codegen::U0GradientFunc;
use super::{CodeGen, codegen::{U0Func, ResidualFunc, CalcOutFunc}, data_layout::DataLayout};




struct JitFunctions<'ctx> {
    set_u0: JitFunction<'ctx, U0Func>,
    residual: JitFunction<'ctx, ResidualFunc>,
    calc_out: JitFunction<'ctx, CalcOutFunc>,
    set_id: JitFunction<'ctx, SetIdFunc>,
    get_dims: JitFunction<'ctx, GetDimsFunc>,
    set_inputs: JitFunction<'ctx, SetInputsFunc>,
    get_out: JitFunction<'ctx, GetOutFunc>,
}

struct JitGradFunctions<'ctx> {
    set_u0_grad: JitFunction<'ctx, U0GradientFunc>,
    residual_grad: JitFunction<'ctx, ResidualGradientFunc>,
    calc_out_grad: JitFunction<'ctx, CalcOutGradientFunc>,
    set_inputs_grad: JitFunction<'ctx, SetInputsGradientFunc>,
}

struct CompilerData<'ctx> {
    codegen: CodeGen<'ctx>,
    jit_functions: JitFunctions<'ctx>,
    jit_grad_functions: JitGradFunctions<'ctx>,
}

#[self_referencing]
pub struct Compiler {
    context: Context,

    #[borrows(context)]
    #[not_covariant]
    data: CompilerData<'this>,
    
    number_of_states: usize,
    number_of_parameters: usize,
    number_of_outputs: usize,
    data_layout: DataLayout,
    output_base_filename: String,
}

impl Compiler {
    const OPT_VARIENTS: [&str; 2] = ["opt-14", "opt"];
    const CLANG_VARIENTS: [&str; 2] = ["clang", "clang-14"];
    fn find_opt() -> Result<&'static str> {
        find_executable(&Compiler::OPT_VARIENTS)
    }
    fn find_clang() -> Result<&'static str> {
        find_executable(&Compiler::CLANG_VARIENTS)
    }
    fn find_enzyme_lib() -> Result<String> {
        let enzyme_lib_varients = ["LLVMEnzyme-14.so", "LLVMEnzyme-14.dylib"];
        find_library_path(&enzyme_lib_varients)
    }
    pub fn from_discrete_model(model: &DiscreteModel, out: &str) -> Result<Self> { 
        let number_of_states = usize::try_from(
            *model.state().shape().first().unwrap_or(&1)
        ).unwrap();
        let input_names = model.inputs().iter().map(|input| input.name().to_owned()).collect::<Vec<_>>();
        let data_layout = DataLayout::new(model);
        let context = Context::create();
        let number_of_parameters = input_names.iter().fold(0, |acc, name| acc + data_layout.get_data_length(name).unwrap());
        let number_of_outputs = data_layout.get_data_length("out").unwrap();
        CompilerTryBuilder {
            data_layout,
            number_of_states,
            number_of_parameters,
            number_of_outputs,
            context,
            output_base_filename: out.to_owned(),
            data_builder: |context| {
                let module = context.create_module(model.name());
                let real_type = context.f64_type();
                let real_type_str = "f64";
                let mut codegen = CodeGen::new(model, &context, module, real_type, real_type_str);
                
                let _set_u0 = codegen.compile_set_u0(model)?;
                let _set_u0_grad = codegen.compile_gradient(_set_u0, &[CompileGradientArgType::Dup, CompileGradientArgType::Const, CompileGradientArgType::Dup, CompileGradientArgType::Dup])?;
                let _residual = codegen.compile_residual(model)?;
                let _residual_grad = codegen.compile_gradient(_residual, &[CompileGradientArgType::Const, CompileGradientArgType::Dup, CompileGradientArgType::Dup, CompileGradientArgType::Dup, CompileGradientArgType::Const, CompileGradientArgType::DupNoNeed])?;
                let _calc_out = codegen.compile_calc_out(model)?;
                let _calc_out_grad = codegen.compile_gradient(_calc_out, &[CompileGradientArgType::Const, CompileGradientArgType::Dup, CompileGradientArgType::Dup, CompileGradientArgType::Dup, CompileGradientArgType::Const])?;
                let _set_id = codegen.compile_set_id(model)?;
                let _get_dims= codegen.compile_get_dims(model)?;
                let _set_inputs = codegen.compile_set_inputs(model)?;
                let _set_inputs_grad = codegen.compile_gradient(_set_inputs, &[CompileGradientArgType::Dup, CompileGradientArgType::Dup])?;
                let _get_output = codegen.compile_get_tensor(model, "out")?;
                

                let pre_enzyme_bitcodefilename = Compiler::get_pre_enzyme_bitcode_filename(out);
                codegen.module().write_bitcode_to_path(Path::new(&pre_enzyme_bitcodefilename));
    
                let opt_name = Compiler::find_opt()?;
                let enzyme_lib = Compiler::find_enzyme_lib()?;

                let bitcodefilename = Compiler::get_bitcode_filename(out);
                let output = Command::new(opt_name)
                    .arg(pre_enzyme_bitcodefilename.as_str())
                    .arg(format!("-load={}", enzyme_lib))
                    .arg("-enzyme")
                    .arg("--enable-new-pm=0")
                    .arg("-o").arg(bitcodefilename.as_str())
                    .output()?;
                
                if let Some(code) = output.status.code() {
                    if code != 0 {
                        println!("{}", String::from_utf8_lossy(&output.stderr));
                        return Err(anyhow!("{} returned error code {}", opt_name, code));
                    }
                }
                
                let buffer = MemoryBuffer::create_from_file(Path::new(bitcodefilename.as_str())).unwrap();
                let module = Module::parse_bitcode_from_buffer(&buffer, context).map_err(|e| anyhow::anyhow!("Error parsing bitcode: {:?}", e))?;
                let ee = module.create_jit_execution_engine(OptimizationLevel::None).map_err(|e| anyhow::anyhow!("Error creating execution engine: {:?}", e))?;

                let set_u0 = Compiler::jit("set_u0", &ee)?;
                let residual = Compiler::jit("residual", &ee)?;
                let calc_out = Compiler::jit("calc_out", &ee)?;
                let set_id = Compiler::jit("set_id", &ee)?;
                let get_dims= Compiler::jit("get_dims", &ee)?;
                let set_inputs = Compiler::jit("set_inputs", &ee)?;
                let get_out= Compiler::jit("get_out", &ee)?;
                
                let set_inputs_grad = Compiler::jit("set_inputs_grad", &ee)?;
                let calc_out_grad = Compiler::jit("calc_out_grad", &ee)?;
                let residual_grad = Compiler::jit("residual_grad", &ee)?;
                let set_u0_grad = Compiler::jit("set_u0_grad", &ee)?;
                
                let data = CompilerData {
                    codegen: codegen,
                    jit_functions: JitFunctions {
                        set_u0,
                        residual,
                        calc_out,
                        set_id,
                        get_dims,
                        set_inputs,
                        get_out,
                    },
                    jit_grad_functions: JitGradFunctions {
                        set_u0_grad,
                        residual_grad,
                        calc_out_grad,
                        set_inputs_grad,
                    },
                };
                Ok(data)
            }
        }.try_build()
    }
    
    pub fn compile(&self, standalone: bool, wasm: bool) -> Result<()> {

        let opt_name = Compiler::find_opt()?;
        let clang_name = Compiler::find_clang()?;
        let enzyme_lib = Compiler::find_enzyme_lib()?;
        let out = self.borrow_output_base_filename();
        let object_filename = Compiler::get_object_filename(out);
        let bitcodefilename = Compiler::get_bitcode_filename(out);
        let mut command = Command::new(clang_name);
        command.arg(bitcodefilename.as_str())
            .arg("-c")
            .arg(format!("-fplugin={}", enzyme_lib))
            .arg("-o").arg(object_filename.as_str());
    
        if wasm {
            command.arg("-target").arg("wasm32-unknown-emscripten");
        }
        
        let output = command.output().unwrap();
        
        
        if let Some(code) = output.status.code() {
            if code != 0 {
                println!("{}", String::from_utf8_lossy(&output.stderr));
                return Err(anyhow!("{} returned error code {}", opt_name, code));
            }
        }
        
        // link the object file and our runtime library
        let mut command = if wasm {
            let emcc_varients = ["emcc"];
            let command_name = find_executable(&emcc_varients)?;
            let exported_functions = vec![
                "Vector_destroy",
                "Vector_create",
                "Vector_create_with_capacity",
                "Vector_push",
                
                "Options_destroy",
                "Options_create",

                "Sundials_destroy",
                "Sundials_create",
                "Sundials_init",
                "Sundials_solve",
            ];
            let mut linked_files = vec![
                "libdiffeq_runtime_lib.a",
                "libsundials_idas.a",
                "libsundials_sunlinsolklu.a",
                "libklu.a",
                "libamd.a",
                "libcolamd.a",
                "libbtf.a",
                "libsuitesparseconfig.a",
                "libsundials_sunmatrixsparse.a",
                "libargparse.a",
            ];
            if standalone {
                linked_files.push("libdiffeq_runtime_wasm.a");
            }
            let linked_files = linked_files;
            let runtime_path = find_runtime_path(&linked_files)?;
            let mut command = Command::new(command_name);
            command.arg("-o").arg(out).arg(object_filename.as_str());
            for file in linked_files {
                command.arg(Path::new(runtime_path.as_str()).join(file));
            }
            if !standalone {
                let exported_functions = exported_functions.into_iter().map(|s| format!("_{}", s)).collect::<Vec<_>>().join(",");
                command.arg("-s").arg(format!("EXPORTED_FUNCTIONS={}", exported_functions));
                command.arg("--no-entry");
            }
            command
        } else {
            let mut command = Command::new(clang_name);
            command.arg("-o").arg(out).arg(object_filename.as_str());
            if standalone {
                command.arg("-ldiffeq_runtime");
            } else {
                command.arg("-ldiffeq_runtime_lib");
            }
            command
        };
        
        let output = command.output();
        
        let output = match output {
            Ok(output) => output,
            Err(e) => {
                let args = command.get_args().map(|s| s.to_str().unwrap()).collect::<Vec<_>>().join(" ");
                println!("{} {}", command.get_program().to_os_string().to_str().unwrap(), args);
                return Err(anyhow!("Error linking in runtime: {}", e));
            }
        };

        if let Some(code) = output.status.code() {
            if code != 0 {
                let args = command.get_args().map(|s| s.to_str().unwrap()).collect::<Vec<_>>().join(" ");
                println!("{} {}", command.get_program().to_os_string().to_str().unwrap(), args);
                println!("{}", String::from_utf8_lossy(&output.stderr));
                return Err(anyhow!("Error linking in runtime, returned error code {}", code));
            }
        }
        Ok(())
    }
    
    fn get_pre_enzyme_bitcode_filename(out: &str) -> String {
        format!("{}.pre-enzyme.bc", out)
    }

    fn get_bitcode_filename(out: &str) -> String {
        format!("{}.bc", out)
    }

    fn get_object_filename(out: &str) -> String {
        format!("{}.o", out)
    }

    fn jit<'ctx, T>(name: &str, ee: &ExecutionEngine<'ctx>) -> Result<JitFunction<'ctx, T>> 
    where T: UnsafeFunctionPointer
    {
        let maybe_fn = unsafe { ee.get_function::<T>(name) };
        let compiled_fn = match maybe_fn {
            Ok(f) => Ok(f),
            Err(err) => {
                Err(anyhow!("Error during jit for {}: {}", name, err))
            },
        };
        compiled_fn
    }

    pub fn get_tensor_data<'a>(&self, name: &str, data: &'a [f64]) -> Option<&'a [f64]> {
        let index = self.borrow_data_layout().get_data_index(name)?;
        let nnz = self.borrow_data_layout().get_data_length(name)?;
        Some(&data[index..index+nnz])
    }

    pub fn set_u0(&self, yy: &mut [f64], yp: &mut [f64], data: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        self.with_data(|compiler| {
            let indices_ptr = self.borrow_data_layout().indices().as_ptr();
            let yy_ptr = yy.as_mut_ptr();
            let yp_ptr = yp.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            unsafe { compiler.jit_functions.set_u0.call(data_ptr, indices_ptr, yy_ptr, yp_ptr); }
        });
        Ok(())
    }

    pub fn set_u0_grad(&self, yy: &mut [f64], dyy: &mut [f64], yp: &mut [f64], dyp: &mut [f64], data: &mut [f64], ddata: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        if dyy.len() != number_of_states {
            return Err(anyhow!("Expected {} states for dyy, got {}", number_of_states, dyy.len()));
        }
        if dyp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives for dyp, got {}", number_of_states, dyp.len()));
        }
        if data.len() != self.data_len() {
            return Err(anyhow!("Expected {} data, got {}", self.data_len(), data.len()));
        }
        if ddata.len() != self.data_len() {
            return Err(anyhow!("Expected {} data for ddata, got {}", self.data_len(), ddata.len()));
        }
        self.with_data(|compiler| {
            let indices_ptr = self.borrow_data_layout().indices().as_ptr();
            let yy_ptr = yy.as_mut_ptr();
            let yp_ptr = yp.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            let dyy_ptr = dyy.as_mut_ptr();
            let dyp_ptr = dyp.as_mut_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            unsafe { compiler.jit_grad_functions.set_u0_grad.call(data_ptr, ddata_ptr, indices_ptr, yy_ptr, dyy_ptr, yp_ptr, dyp_ptr); }
        });
        Ok(())
    }

    pub fn residual(&self, t: f64, yy: &[f64], yp: &[f64], data: &mut [f64], rr: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        if rr.len() != number_of_states {
            return Err(anyhow!("Expected {} residual states, got {}", number_of_states, rr.len()));
        }
        if data.len() != self.data_len() {
            return Err(anyhow!("Expected {} data, got {}", self.data_len(), data.len()));
        }
        self.with_data(|compiler| {
            let layout = self.borrow_data_layout();
            let indices_ptr = layout.indices().as_ptr();
            let yy_ptr = yy.as_ptr();
            let yp_ptr = yp.as_ptr();
            let rr_ptr = rr.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            unsafe { compiler.jit_functions.residual.call(t, yy_ptr, yp_ptr, data_ptr, indices_ptr, rr_ptr); }
        });
        Ok(())
    }

    pub fn data_len(&self) -> usize {
        self.with(|compiler| {
            compiler.data_layout.data().len()
        })
    }

    pub fn get_new_data(&self) -> Vec<f64> {
        vec![0.; self.data_len()]
    }

    pub fn residual_grad(&self, t: f64, yy: &[f64], dyy: &[f64], yp: &[f64], dyp: &[f64], data: &mut [f64], ddata: &mut [f64], rr: &mut [f64], drr: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        if rr.len() != number_of_states {
            return Err(anyhow!("Expected {} residual states, got {}", number_of_states, rr.len()));
        }
        if dyy.len() != number_of_states {
            return Err(anyhow!("Expected {} states for dyy, got {}", number_of_states, dyy.len()));
        }
        if dyp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives for dyp, got {}", number_of_states, dyp.len()));
        }
        if drr.len() != number_of_states {
            return Err(anyhow!("Expected {} residual states for drr, got {}", number_of_states, drr.len()));
        }
        if data.len() != self.data_len() {
            return Err(anyhow!("Expected {} data, got {}", self.data_len(), data.len()));
        }
        if ddata.len() != self.data_len() {
            return Err(anyhow!("Expected {} data for ddata, got {}", self.data_len(), ddata.len()));
        }
        self.with_data(|compiler| {
            let layout = self.borrow_data_layout();
            let indices_ptr = layout.indices().as_ptr();
            let yy_ptr = yy.as_ptr();
            let yp_ptr = yp.as_ptr();
            let rr_ptr = rr.as_mut_ptr();
            let dyy_ptr = dyy.as_ptr();
            let dyp_ptr = dyp.as_ptr();
            let drr_ptr = drr.as_mut_ptr();
            let data_ptr = data.as_mut_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            unsafe { compiler.jit_grad_functions.residual_grad.call(t, yy_ptr, dyy_ptr, yp_ptr, dyp_ptr, data_ptr, ddata_ptr, indices_ptr, rr_ptr, drr_ptr); }
        });
        Ok(())
    }

    pub fn calc_out(&self, t: f64, yy: &[f64], yp: &[f64], data: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        if data.len() != self.data_len() {
            return Err(anyhow!("Expected {} data, got {}", self.data_len(), data.len()));
        }
        self.with_data(|compiler| {
            let layout = self.borrow_data_layout();
            let indices_ptr = layout.indices().as_ptr();
            let yy_ptr = yy.as_ptr();
            let yp_ptr = yp.as_ptr();
            let data_ptr = data.as_mut_ptr();
            unsafe { compiler.jit_functions.calc_out.call(t, yy_ptr, yp_ptr, data_ptr, indices_ptr); }
        });
        Ok(())
    }

    pub fn calc_out_grad(&self, t: f64, yy: &[f64], dyy: &[f64], yp: &[f64], dyp: &[f64], data: &mut [f64], ddata: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        if data.len() != self.data_len() {
            return Err(anyhow!("Expected {} data, got {}", self.data_len(), data.len()));
        }
        if dyy.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} states for dyy, got {}", number_of_states, dyy.len()));
        }
        if dyp.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} state derivatives for dyp, got {}", number_of_states, dyp.len()));
        }
        if ddata.len() != self.data_len() {
            return Err(anyhow!("Expected {} data for ddata, got {}", self.data_len(), ddata.len()));
        }
        self.with_data(|compiler| {
            let layout = self.borrow_data_layout();
            let indices_ptr = layout.indices().as_ptr();
            let yy_ptr = yy.as_ptr();
            let yp_ptr = yp.as_ptr();
            let data_ptr = data.as_mut_ptr();
            let dyy_ptr = dyy.as_ptr();
            let dyp_ptr = dyp.as_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            unsafe { compiler.jit_grad_functions.calc_out_grad.call(t, yy_ptr, dyy_ptr, yp_ptr, dyp_ptr, data_ptr, ddata_ptr, indices_ptr); }
        });
        Ok(())
    }

    /// Get various dimensions of the model
    /// 
    /// # Returns
    /// 
    /// A tuple of the form `(n_states, n_inputs, n_outputs, n_data, n_indices)`
    pub fn get_dims(&self) -> (usize, usize, usize, usize, usize) {
        let mut n_states = 0u32;
        let mut n_inputs= 0u32;
        let mut n_outputs = 0u32;
        let mut n_data= 0u32;
        let mut n_indices = 0u32;
        self.with(|compiler| {
            unsafe { compiler.data.jit_functions.get_dims.call(&mut n_states, &mut n_inputs, &mut n_outputs, &mut n_data, &mut n_indices); }
        });
        (n_states as usize, n_inputs as usize, n_outputs as usize, n_data as usize, n_indices as usize)
    }

    pub fn set_inputs(&self, inputs: &[f64], data: &mut [f64]) -> Result<()> {
        let (_, n_inputs, _, _, _) = self.get_dims();
        if n_inputs != inputs.len() {
            return Err(anyhow!("Expected {} inputs, got {}", n_inputs, inputs.len()));
        }
        if data.len() != self.data_len() {
            return Err(anyhow!("Expected {} data, got {}", self.data_len(), data.len()));
        }
        self.with_data(|compiler| {
            let data_ptr = data.as_mut_ptr();
            unsafe { compiler.jit_functions.set_inputs.call(inputs.as_ptr(), data_ptr); }
        });
        Ok(())
    }

    pub fn set_inputs_grad(&self, inputs: &[f64], dinputs: &[f64], data: &mut [f64], ddata: &mut [f64]) -> Result<()> {
        let (_, n_inputs, _, _, _) = self.get_dims();
        if n_inputs != inputs.len() {
            return Err(anyhow!("Expected {} inputs, got {}", n_inputs, inputs.len()));
        }
        if data.len() != self.data_len() {
            return Err(anyhow!("Expected {} data, got {}", self.data_len(), data.len()));
        }
        if dinputs.len() != n_inputs {
            return Err(anyhow!("Expected {} inputs for dinputs, got {}", n_inputs, dinputs.len()));
        }
        if ddata.len() != self.data_len() {
            return Err(anyhow!("Expected {} data for ddata, got {}", self.data_len(), ddata.len()));
        }
        self.with_data(|compiler| {
            let data_ptr = data.as_mut_ptr();
            let ddata_ptr = ddata.as_mut_ptr();
            let dinputs_ptr = dinputs.as_ptr();
            unsafe { compiler.jit_grad_functions.set_inputs_grad.call(inputs.as_ptr(), dinputs_ptr, data_ptr, ddata_ptr); }
        });
        Ok(())
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
            unsafe { compiler.data.jit_functions.get_out.call(data_ptr, tensor_data_ptr_ptr, tensor_data_len_ptr); }
        });
        assert!(tensor_data_len as usize == n_outputs);
        unsafe { std::slice::from_raw_parts(tensor_data_ptr, tensor_data_len as usize) }
    }

    pub fn set_id(&self, id: &mut [f64]) -> Result<()> {
        let (n_states, _, _, _, _) = self.get_dims();
        if n_states != id.len() {
            return Err(anyhow!("Expected {} states, got {}", n_states, id.len()));
        }
        self.with_data(|compiler| {
            Ok(unsafe { compiler.jit_functions.set_id.call(id.as_mut_ptr()); })
        })
    }

    fn get_native_machine() -> Result<TargetMachine> {
        Target::initialize_native(&InitializationConfig::default()).map_err(|e| anyhow!("{}", e))?;
        let opt = OptimizationLevel::Default;
        let reloc = RelocMode::Default;
        let model = CodeModel::Default;
        let target_triple = TargetMachine::get_default_triple();
        let target  = Target::from_triple(&target_triple).unwrap();
        let target_machine = target.create_target_machine(
            &target_triple,
            TargetMachine::get_host_cpu_name().to_str().unwrap(),
            TargetMachine::get_host_cpu_features().to_str().unwrap(),
            opt,
            reloc,
            model
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
        let target_machine = target.create_target_machine(
            &target_triple,
            "generic",
            "",
            opt,
            reloc,
            model
        )
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
        let target_machine = Compiler::get_native_machine()?;
        self.with_data(|data|
                target_machine.write_to_file(data.codegen.module(), FileType::Object, &path).map_err(|e| anyhow::anyhow!("Error writing object file: {:?}", e))
        )
    }

    pub fn write_wasm_object_file(&self, path: &Path) -> Result<()> {
        let target_machine = Compiler::get_wasm_machine()?;
        self.with_data(|data|
                target_machine.write_to_file(data.codegen.module(), FileType::Object, &path).map_err(|e| anyhow::anyhow!("Error writing object file: {:?}", e))
        )
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


#[cfg(test)]
mod tests {
    use crate::{parser::{parse_ds_string, parse_ms_string}, continuous::ModelInfo};
    use approx::assert_relative_eq;

    use super::*;

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
        let object = Compiler::from_discrete_model(&discrete_model, "test_output/compiler_test_object_file").unwrap();
        let path = Path::new("main.o");
        object.write_object_file(path).unwrap();
    }

    fn tensor_test_common(text: &str, tmp_loc: &str, tensor_name: &str) -> Vec<Vec<f64>> {
        let full_text = format!("
            {}
        ", text);
        let model = parse_ds_string(full_text.as_str()).unwrap();
        let discrete_model = match DiscreteModel::build("$name", &model) {
            Ok(model) => {
                model
            }
            Err(e) => {
                panic!("{}", e.as_error_message(full_text.as_str()));
            }
        };
        let compiler = Compiler::from_discrete_model(&discrete_model, tmp_loc).unwrap();
        let mut u0 = vec![1.];
        let mut up0 = vec![1.];
        let mut res = vec![0.];
        let mut data = compiler.get_new_data();
        let mut grad_data = Vec::new();
        let (_n_states, n_inputs, _n_outputs, _n_data, _n_indices) = compiler.get_dims();
        for _ in 0..n_inputs {
            grad_data.push(compiler.get_new_data());
        }
        let mut results = Vec::new();
        let inputs = vec![1.; n_inputs];
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice()).unwrap();
        compiler.set_u0(u0.as_mut_slice(), up0.as_mut_slice(), data.as_mut_slice()).unwrap();
        compiler.residual(0., u0.as_slice(), up0.as_slice(), data.as_mut_slice(), res.as_mut_slice()).unwrap();
        compiler.calc_out(0., u0.as_slice(), up0.as_slice(), data.as_mut_slice()).unwrap();
        results.push(compiler.get_tensor_data(tensor_name, data.as_slice()).unwrap().to_vec());
        for i in 0..n_inputs {
            let mut dinputs = vec![0.; n_inputs];
            dinputs[i] = 1.0;
            let mut ddata = compiler.get_new_data();
            let mut du0 = vec![0.];
            let mut dup0 = vec![0.];
            let mut dres = vec![0.];
            compiler.set_inputs_grad(inputs.as_slice(), dinputs.as_slice(), grad_data[i].as_mut_slice(), ddata.as_mut_slice()).unwrap();
            compiler.set_u0_grad(u0.as_mut_slice(), du0.as_mut_slice(), up0.as_mut_slice(), dup0.as_mut_slice(), grad_data[i].as_mut_slice(), ddata.as_mut_slice()).unwrap();
            compiler.residual_grad(0., u0.as_slice(), du0.as_slice(), up0.as_slice(), dup0.as_slice(), grad_data[i].as_mut_slice(), ddata.as_mut_slice(), res.as_mut_slice(), dres.as_mut_slice()).unwrap();
            compiler.calc_out_grad(0., u0.as_slice(), du0.as_slice(), up0.as_slice(), dup0.as_slice(), grad_data[i].as_mut_slice(), ddata.as_mut_slice()).unwrap();
            results.push(compiler.get_tensor_data(tensor_name, ddata.as_slice()).unwrap().to_vec());
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
                    dudt_i {{
                        dydt = 0,
                    }}
                    F_i {{
                        dydt,
                    }}
                    G_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", $text);
                let tmp_loc = format!("test_output/compiler_tensor_test_{}", stringify!($name));
                let results = tensor_test_common(full_text.as_str(), tmp_loc.as_str(), $tensor_name);
                assert_relative_eq!(results[0].as_slice(), $expected_value.as_slice());
            }
        )*
        }
    }

    tensor_test!{
        exp_function: "r { exp(2) }" expect "r" vec![f64::exp(2.0)],
        exp_function_time: "r { exp(t) }" expect "r" vec![f64::exp(0.0)],
        sigmoid_function: "r { sigmoid(0.1) }" expect "r" vec![1.0 / (1.0 + f64::exp(-0.1))],
        scalar: "r {2}" expect "r" vec![2.0,],
        constant: "r_i {2, 3}" expect "r" vec![2., 3.],
        expression: "r_i {2 + 3, 3 * 2}" expect "r" vec![5., 6.],
        derived: "r_i {2, 3} k_i { 2 * r_i }" expect "k" vec![4., 6.],
        concatenate: "r_i {2, 3} k_i { r_i, 2 * r_i }" expect "k" vec![2., 3., 4., 6.],
        ones_matrix_dense: "I_ij { (0:2, 0:2): 1 }" expect "I" vec![1., 1., 1., 1.],
        dense_matrix: "A_ij { (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 }" expect "A" vec![1., 2., 3., 4.], 
        identity_matrix_diagonal: "I_ij { (0..2, 0..2): 1 }" expect "I" vec![1., 1.],
        concatenate_diagonal: "A_ij { (0..2, 0..2): 1 } B_ij { (0:2, 0:2): A_ij, (2:4, 2:4): A_ij }" expect "B" vec![1., 1., 1., 1.],
        identity_matrix_sparse: "I_ij { (0, 0): 1, (1, 1): 2 }" expect "I" vec![1., 2.],
        concatenate_sparse: "A_ij { (0, 0): 1, (1, 1): 2 } B_ij { (0:2, 0:2): A_ij, (2:4, 2:4): A_ij }" expect "B" vec![1., 2., 1., 2.],
        sparse_rearrange: "A_ij { (0, 0): 1, (1, 1): 2, (0, 1): 3 }" expect "A" vec![1., 3., 2.],
        sparse_expression: "A_ij { (0, 0): 1, (0, 1): 2, (1, 1): 3 } B_ij { 2 * A_ij }" expect "B" vec![2., 4., 6.],
        sparse_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 0): 2, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" vec![1., 8.],
        diag_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" vec![1., 6.],
        dense_matrix_vect_multiply: "A_ij {  (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" vec![5., 11.],
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
                    F_i {{
                        dydt,
                    }}
                    G_i {{
                        y,
                    }}
                    out_i {{
                        y,
                    }}
                ", $text);
                let tmp_loc = format!("test_output/compiler_tensor_grad_test_{}", stringify!($name));
                let results = tensor_test_common(full_text.as_str(), tmp_loc.as_str(), $tensor_name);
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
    fn test_repeated_grad() {
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
            F_i {
                dydt,
            }
            G_i {
                r,
            }
            out_i {
                y,
            }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = match DiscreteModel::build("test_repeated_grad", &model) {
            Ok(model) => {
                model
            }
            Err(e) => {
                panic!("{}", e.as_error_message(full_text));
            }
        };
        let compiler = Compiler::from_discrete_model(&discrete_model, "test_output/compiler_test_repeated_grad").unwrap();
        let mut u0 = vec![1.];
        let mut up0 = vec![1.];
        let mut du0 = vec![1.];
        let mut dup0 = vec![1.];
        let mut res = vec![0.];
        let mut dres = vec![0.];
        let mut data = compiler.get_new_data();
        let mut ddata = compiler.get_new_data();
        let (_n_states, n_inputs, _n_outputs, _n_data, _n_indices) = compiler.get_dims();

        for _ in 0..3 {
            let inputs = vec![2.; n_inputs];
            let dinputs = vec![1.; n_inputs];
            compiler.set_inputs_grad(inputs.as_slice(), dinputs.as_slice(), data.as_mut_slice(), ddata.as_mut_slice()).unwrap();
            compiler.set_u0_grad(u0.as_mut_slice(), du0.as_mut_slice(), up0.as_mut_slice(), dup0.as_mut_slice(), data.as_mut_slice(), ddata.as_mut_slice()).unwrap();
            compiler.residual_grad(0., u0.as_slice(), du0.as_slice(), up0.as_slice(), dup0.as_slice(), data.as_mut_slice(), ddata.as_mut_slice(), res.as_mut_slice(), dres.as_mut_slice()).unwrap();
            assert_relative_eq!(dres.as_slice(), vec![-8.].as_slice());
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
            F_i {
                dydt,
                0,
            }
            G_i {
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
        let compiler = Compiler::from_discrete_model(&discrete_model, "test_output/compiler_test_additional_functions").unwrap();
        let (n_states, n_inputs, n_outputs, n_data, n_indices) = compiler.get_dims();
        assert_eq!(n_states, 2);
        assert_eq!(n_inputs, 1);
        assert_eq!(n_outputs, 3);
        assert_eq!(n_data, compiler.data_len());
        assert_eq!(n_indices, compiler.borrow_data_layout().indices().len());

        let mut data = compiler.get_new_data();
        let inputs = vec![1.1];
        compiler.set_inputs(inputs.as_slice(), data.as_mut_slice()).unwrap();
        
        let inputs = compiler.get_tensor_data("k", data.as_slice()).unwrap();
        assert_relative_eq!(inputs, vec![1.1].as_slice());

        let mut id = vec![0.0, 0.0];
        compiler.set_id(id.as_mut_slice()).unwrap();
        assert_eq!(id, vec![1.0, 0.0]);

        let mut u = vec![0., 0.];
        let mut up = vec![0., 0.];
        compiler.set_u0(u.as_mut_slice(), up.as_mut_slice(), data.as_mut_slice()).unwrap();
        assert_relative_eq!(u.as_slice(), vec![1., 2.].as_slice());
        assert_relative_eq!(up.as_slice(), vec![0., 0.].as_slice());

        let mut rr = vec![1., 1.];
        compiler.residual(0., u.as_slice(), up.as_slice(), data.as_mut_slice(), rr.as_mut_slice()).unwrap();
        assert_relative_eq!(rr.as_slice(), vec![0., 0.].as_slice());

        compiler.calc_out(0., u.as_slice(), up.as_slice(), data.as_mut_slice()).unwrap();
        let out = compiler.get_out(data.as_slice());
        assert_relative_eq!(out, vec![1., 2., 4.].as_slice());
    }
}