use std::path::Path;
use anyhow::anyhow;

use anyhow::Result;
use inkwell::{context::Context, OptimizationLevel, targets::{TargetTriple, InitializationConfig, Target, RelocMode, CodeModel, FileType}, execution_engine::{JitFunction, ExecutionEngine, UnsafeFunctionPointer}};
use ouroboros::self_referencing;
use crate::discretise::DiscreteModel;


use super::{CodeGen, codegen::{U0Func, ResidualFunc, CalcOutFunc}, data_layout::DataLayout};

struct CompilerData<'ctx> {
    codegen: CodeGen<'ctx>,
    set_u0: JitFunction<'ctx, U0Func>,
    residual: JitFunction<'ctx, ResidualFunc>,
    calc_out: JitFunction<'ctx, CalcOutFunc>,
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

                let set_u0 = codegen.compile_set_u0(model)?;
                let residual = codegen.compile_residual(model)?;
                let calc_out = codegen.compile_calc_out(model)?;

                set_u0.print_to_stderr();
                residual.print_to_stderr();
                calc_out.print_to_stderr();

                let set_u0 = Compiler::jit("set_u0", &ee)?;
                let residual = Compiler::jit("residual", &ee)?;
                let calc_out = Compiler::jit("calc_out", &ee)?;

                Ok({
                    CompilerData {
                        codegen,
                        set_u0,
                        residual,
                        calc_out,
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

    pub fn set_inputs(&mut self, inputs: &[f64]) -> Result<()> {
        let number_of_inputs = self.borrow_input_names().len();
        if number_of_inputs != inputs.len() {
            return Err(anyhow!("Expected {} inputs, got {}", number_of_inputs, inputs.len()));
        }
        self.with_mut(|compiler| {
            let layout = compiler.data_layout;
            let mut curr_index = 0;
            for name in compiler.input_names.iter() {
                let data = layout.get_tensor_data_mut(name).unwrap();
                data.copy_from_slice(&inputs[curr_index..curr_index + data.len()]);
                curr_index += data.len();
            }
        });
        
        Ok(())
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

    pub fn write_object_file(&self) -> Result<()> {
        Target::initialize_x86(&InitializationConfig::default());

        let opt = OptimizationLevel::Default;
        let reloc = RelocMode::Default;
        let model = CodeModel::Default;
        let target = Target::from_name("x86-64").unwrap();
        let target_machine = target.create_target_machine(
            &TargetTriple::create("x86_64-pc-linux-gnu"),
            "x86-64",
            "+avx2",
            opt,
            reloc,
            model
        )
        .unwrap();

        let path = Path::new("main.o");
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
    use crate::parser::parse_ds_string;
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_object_file() {
        let full_text = "
            u_i {
                y = 1,
            }
            dudt_i {
                dydt = 0,
            }
            F_i {
                dydt,
            }
            G_i {
                y,
            }
            out_i {
                y,
            }
        ";
        let model = parse_ds_string(full_text).unwrap();
        let discrete_model = DiscreteModel::build("$name", &model).unwrap();
        let object = Compiler::from_discrete_model(&discrete_model).unwrap();
        object.write_object_file().unwrap();

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
                compiler.set_inputs(inputs.as_slice()).unwrap();
                compiler.set_u0(u0.as_mut_slice(), up0.as_mut_slice()).unwrap();
                let tensor = compiler.get_tensor_data($tensor_name).unwrap();
                assert_relative_eq!(tensor, $expected_value.as_slice());
            }
        )*
        }
    }

    tensor_test!{
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
}