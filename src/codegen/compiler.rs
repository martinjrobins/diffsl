use std::path::Path;
use anyhow::anyhow;

use anyhow::Result;
use inkwell::targets::TargetMachine;
use inkwell::{context::Context, OptimizationLevel, targets::{TargetTriple, InitializationConfig, Target, RelocMode, CodeModel, FileType}, execution_engine::{JitFunction, ExecutionEngine, UnsafeFunctionPointer}};
use ouroboros::self_referencing;
use crate::discretise::DiscreteModel;


use super::codegen::GetDimsFunc;
use super::codegen::GetOutFunc;
use super::codegen::SetIdFunc;
use super::codegen::SetInputsFunc;
use super::{CodeGen, codegen::{U0Func, ResidualFunc, CalcOutFunc}, data_layout::DataLayout};

struct CompilerData<'ctx> {
    codegen: CodeGen<'ctx>,
    set_u0: JitFunction<'ctx, U0Func>,
    residual: JitFunction<'ctx, ResidualFunc>,
    calc_out: JitFunction<'ctx, CalcOutFunc>,
    set_id: JitFunction<'ctx, SetIdFunc>,
    get_dims: JitFunction<'ctx, GetDimsFunc>,
    set_inputs: JitFunction<'ctx, SetInputsFunc>,
    get_out: JitFunction<'ctx, GetOutFunc>,

}

#[self_referencing]
pub struct Compiler {
    context: Context,
    #[borrows(context)]
    #[not_covariant]
    data: CompilerData<'this>,

    data_layout: DataLayout,
    number_of_states: usize,
    input_names: Vec<String>,
}

impl Compiler {
    pub fn from_discrete_model(model: &DiscreteModel) -> Result<Self> { 
        let number_of_states = usize::try_from(
            *model.state().shape().first().unwrap_or(&1)
        ).unwrap();
        let input_names = model.inputs().iter().map(|input| input.name().to_owned()).collect::<Vec<_>>();
        let data_layout = DataLayout::new(model);
        let context = Context::create();
        CompilerTryBuilder {
            context,
            data_layout,
            number_of_states,
            input_names,
            data_builder: |context| {
                let module = context.create_module(model.name());

                let real_type = context.f64_type();
                let real_type_str = "f64";
                let ee = module.create_jit_execution_engine(OptimizationLevel::None).map_err(|e| anyhow::anyhow!("Error creating execution engine: {:?}", e))?;
                let mut codegen = CodeGen::new(model, &context, module, real_type, real_type_str);

                let _set_u0 = codegen.compile_set_u0(model)?;
                let _residual = codegen.compile_residual(model)?;
                let _calc_out = codegen.compile_calc_out(model)?;
                let _set_id = codegen.compile_set_id(model)?;
                let _get_dims= codegen.compile_get_dims(model)?;
                let _set_inputs = codegen.compile_set_inputs(model)?;
                let _get_output = codegen.compile_get_tensor(model, "out")?;


                let set_u0 = Compiler::jit("set_u0", &ee)?;
                let residual = Compiler::jit("residual", &ee)?;
                let calc_out = Compiler::jit("calc_out", &ee)?;
                let set_id = Compiler::jit("set_id", &ee)?;
                let get_dims= Compiler::jit("get_dims", &ee)?;
                let set_inputs = Compiler::jit("set_inputs", &ee)?;
                let get_out= Compiler::jit("get_out", &ee)?;


                Ok({
                    CompilerData {
                        codegen,
                        set_u0,
                        residual,
                        calc_out,
                        set_id,
                        get_dims,
                        set_inputs,
                        get_out,
                    }
                })
            }
        }.try_build()
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

    pub fn get_tensor_data(&self, name: &str) -> Option<&[f64]> {
        self.borrow_data_layout().get_tensor_data(name)
    }

    pub fn set_u0(&mut self, yy: &mut [f64], yp: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        self.with_mut(|compiler| {
            let data_ptr = compiler.data_layout.data_mut().as_mut_ptr();
            let indices_ptr = compiler.data_layout.indices().as_ptr();
            let yy_ptr = yy.as_mut_ptr();
            let yp_ptr = yp.as_mut_ptr();
            unsafe { compiler.data.set_u0.call(data_ptr, indices_ptr, yy_ptr, yp_ptr); }
        });
        
        Ok(())
    }

    pub fn residual(&mut self, t: f64, yy: &[f64], yp: &[f64], rr: &mut [f64]) -> Result<()> {
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
        self.with_mut(|compiler| {
            let layout = compiler.data_layout;
            let data_ptr = layout.data_mut().as_mut_ptr();
            let indices_ptr = layout.indices().as_ptr();
            let yy_ptr = yy.as_ptr();
            let yp_ptr = yp.as_ptr();
            let rr_ptr = rr.as_mut_ptr();
            unsafe { compiler.data.residual.call(t, yy_ptr, yp_ptr, data_ptr, indices_ptr, rr_ptr); }
        });
        Ok(())
    }

    pub fn calc_out(&mut self, t: f64, yy: &[f64], yp: &[f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != *self.borrow_number_of_states() {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        self.with_mut(|compiler| {
            let layout = compiler.data_layout;
            let data_ptr = layout.data_mut().as_mut_ptr();
            let indices_ptr = layout.indices().as_ptr();
            let yy_ptr = yy.as_ptr();
            let yp_ptr = yp.as_ptr();
            unsafe { compiler.data.calc_out.call(t, yy_ptr, yp_ptr, data_ptr, indices_ptr); }
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
            unsafe { compiler.data.get_dims.call(&mut n_states, &mut n_inputs, &mut n_outputs, &mut n_data, &mut n_indices); }
        });
        (n_states as usize, n_inputs as usize, n_outputs as usize, n_data as usize, n_indices as usize)
    }

    pub fn set_inputs(&mut self, inputs: &[f64]) -> Result<()> {
        let (_, n_inputs, _, _, _) = self.get_dims();
        if n_inputs != inputs.len() {
            return Err(anyhow!("Expected {} inputs, got {}", n_inputs, inputs.len()));
        }
        self.with_mut(|compiler| {
            let layout = compiler.data_layout;
            let data_ptr = layout.data_mut().as_mut_ptr();
            unsafe { compiler.data.set_inputs.call(inputs.as_ptr(), data_ptr); }
        });
        Ok(())
    }

    pub fn get_out(&self) -> &[f64] {
        let (_, _, n_outputs, _, _) = self.get_dims();
        let mut tensor_data_ptr: *mut f64 = std::ptr::null_mut();
        let mut tensor_data_len = 0u32;
        let tensor_data_ptr_ptr: *mut *mut f64 = &mut tensor_data_ptr;
        let tensor_data_len_ptr: *mut u32 = &mut tensor_data_len;
        self.with(|compiler| {
            let layout = compiler.data_layout;
            let data_ptr = layout.data().as_ptr();
            unsafe { compiler.data.get_out.call(data_ptr, tensor_data_ptr_ptr, tensor_data_len_ptr); }
        });
        assert!(tensor_data_len as usize == n_outputs);
        unsafe { std::slice::from_raw_parts(tensor_data_ptr, tensor_data_len as usize) }
    }

    pub fn set_id(&mut self, id: &mut [f64]) -> Result<()> {
        let (n_states, _, _, _, _) = self.get_dims();
        if n_states != id.len() {
            return Err(anyhow!("Expected {} states, got {}", n_states, id.len()));
        }
        self.with_mut(|compiler| {
            Ok(unsafe { compiler.data.set_id.call(id.as_mut_ptr()); })
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
        self.borrow_input_names().iter().fold(0, |acc, name| acc + self.borrow_data_layout().get_data_length(name).unwrap())
    }

    pub fn number_of_outputs(&self) -> usize {
        self.borrow_data_layout().get_data_length("out")
            .expect("Output data not found")
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
        let object = Compiler::from_discrete_model(&discrete_model).unwrap();
        let path = Path::new("main.o");
        object.write_object_file(path).unwrap();
    }

    macro_rules! tensor_test {
        ($($name:ident: $text:literal expect $tensor_name:literal $expected_value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let text = $text;
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
                let mut compiler = Compiler::from_discrete_model(&discrete_model).unwrap();
                let inputs = vec![];
                let mut u0 = vec![1.];
                let mut up0 = vec![1.];
                let mut res = vec![0.];
                compiler.set_inputs(inputs.as_slice()).unwrap();
                compiler.set_u0(u0.as_mut_slice(), up0.as_mut_slice()).unwrap();
                compiler.residual(0., u0.as_slice(), up0.as_slice(), res.as_mut_slice()).unwrap();
                let tensor = compiler.get_tensor_data($tensor_name).unwrap();
                assert_relative_eq!(tensor, $expected_value.as_slice());
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
        let mut compiler = Compiler::from_discrete_model(&discrete_model).unwrap();
        let (n_states, n_inputs, n_outputs, n_data, n_indices) = compiler.get_dims();
        assert_eq!(n_states, 2);
        assert_eq!(n_inputs, 1);
        assert_eq!(n_outputs, 3);
        assert_eq!(n_data, compiler.borrow_data_layout().data().len());
        assert_eq!(n_indices, compiler.borrow_data_layout().indices().len());

        let inputs = vec![1.1];
        compiler.set_inputs(inputs.as_slice()).unwrap();
        let inputs = compiler.borrow_data_layout().get_tensor_data("k").unwrap();
        assert_relative_eq!(inputs, vec![1.1].as_slice());

        let mut id = vec![0.0, 0.0];
        compiler.set_id(id.as_mut_slice()).unwrap();
        assert_eq!(id, vec![1.0, 0.0]);

        let mut u = vec![0., 0.];
        let mut up = vec![0., 0.];
        compiler.set_u0(u.as_mut_slice(), up.as_mut_slice()).unwrap();
        assert_relative_eq!(u.as_slice(), vec![1., 2.].as_slice());
        assert_relative_eq!(up.as_slice(), vec![0., 0.].as_slice());

        let mut rr = vec![1., 1.];
        compiler.residual(0., u.as_slice(), up.as_slice(), rr.as_mut_slice()).unwrap();
        assert_relative_eq!(rr.as_slice(), vec![0., 0.].as_slice());

        compiler.calc_out(0., u.as_slice(), up.as_slice()).unwrap();
        let out = compiler.get_out();
        assert_relative_eq!(out, vec![1., 2., 4.].as_slice());
    }
}