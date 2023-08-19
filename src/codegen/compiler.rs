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
        let number_of_parameters = model.inputs().iter().fold(0, |acc, input| acc + i64::try_from(input.nnz()).unwrap());
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
                let mut codegen = CodeGen::new(model, &context, module, real_type, real_type_str);

                let set_u0 = codegen.compile_set_u0(model)?;
                let residual = codegen.compile_residual(model)?;
                let calc_out = codegen.compile_calc_out(model)?;

                set_u0.print_to_stderr();
                residual.print_to_stderr();
                calc_out.print_to_stderr();

                let ee = module.create_jit_execution_engine(OptimizationLevel::None).map_err(|e| anyhow::anyhow!("Error creating execution engine: {:?}", e))?;
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

    pub fn set_inputs(&self, inputs: &[f64]) -> Result<()> {
        let layout = self.borrow_data_layout();
        let number_of_inputs = self.borrow_input_names().len();
        if number_of_inputs != inputs.len() {
            return Err(anyhow!("Expected {} inputs, got {}", number_of_inputs, inputs.len()));
        }
        let mut curr_index = 0;
        for name in self.borrow_input_names().iter() {
            let data = layout.get_tensor_data_mut(name).unwrap();
            data.copy_from_slice(&inputs[curr_index..curr_index + data.len()]);
            curr_index += data.len();
        }
        Ok(())
    }

    pub fn set_u0(&self, yy: &mut [f64], yp: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            return Err(anyhow!("Expected {} states, got {}", number_of_states, yy.len()));
        }
        if yp.len() != number_of_states {
            return Err(anyhow!("Expected {} state derivatives, got {}", number_of_states, yp.len()));
        }
        let layout = self.borrow_data_layout();
        let data_ptr = layout.data().as_mut_ptr();
        let indices_ptr = layout.indices().as_ptr();
        let yy_ptr = yy.as_mut_ptr();
        let yp_ptr = yp.as_mut_ptr();
        self.with_data(|data| {
            unsafe { data.set_u0.call(data_ptr, indices_ptr, yy_ptr, yp_ptr); }
        });
        Ok(())
    }

    pub fn residual(&self, t: f64, yy: &[f64], yp: &[f64], rr: &mut [f64]) -> Result<()> {
        let number_of_states = *self.borrow_number_of_states();
        if yy.len() != number_of_states {
            panic!("Expected {} states, got {}", number_of_states, yy.len());
        }
        if yp.len() != number_of_states {
            panic!("Expected {} state derivatives, got {}", number_of_states, yp.len());
        }
        if rr.len() != number_of_states {
            panic!("Expected {} residual states, got {}", number_of_states, rr.len());
        }
        let layout = self.borrow_data_layout();
        let data_ptr = layout.data().as_mut_ptr();
        let indices_ptr = layout.indices().as_ptr();
        let yy_ptr = yy.as_mut_ptr();
        let yp_ptr = yp.as_mut_ptr();
        let rr_ptr = rr.as_mut_ptr();
        self.with_data(|data| {
            unsafe { data.residual.call(t, yy_ptr, yp_ptr, data_ptr, indices_ptr, rr_ptr); }
        });
        Ok(())
    }

    pub fn calc_out(&self, t: f64, yy: &[f64], yp: &[f64], out: &mut [f64]) {
        let layout = self.borrow_data_layout();
        let data_ptr = layout.data().as_mut_ptr();
        let indices_ptr = layout.indices().as_ptr();
        let yy_ptr = yy.as_mut_ptr();
        let yp_ptr = yp.as_mut_ptr();
        self.with_data(|data| {
            unsafe { data.calc_out.call(t, yy_ptr, yp_ptr, data_ptr, indices_ptr); }
        });
        out.copy_from_slice(layout.get_tensor_data("out").unwrap());
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
        self.borrow_input_names().fold(0, |acc, name| acc + self.borrow_data_layout().get_data_length(name).unwrap())
    }


}


#[cfg(test)]
mod tests {
    use crate::parser::parse_ds_string;

    use super::*;

    #[test]
    fn test_wasm() {
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
}