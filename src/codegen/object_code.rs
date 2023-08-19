use std::path::Path;

use anyhow::Result;
use inkwell::{context::Context, passes::PassManager, OptimizationLevel, targets::{TargetTriple, InitializationConfig, Target, RelocMode, CodeModel, FileType}, execution_engine::JitFunction};
use ouroboros::self_referencing;
use crate::discretise::DiscreteModel;


use super::{CodeGen, codegen::{U0Func, ResidualFunc, CalcOutFunc}};

#[self_referencing]
pub struct ObjectCode {
    context: Context,
    #[borrows(context)]
    #[not_covariant]
    codegen: CodeGen<'this>,
}

impl ObjectCode {
    pub fn from_discrete_model(model: &DiscreteModel) -> Result<Self> { 
        ObjectCodeTryBuilder {
            context: Context::create(),
            codegen_builder: |context| {
                let module = context.create_module(model.name());
                let fpm = PassManager::create(&module);
                fpm.add_instruction_combining_pass();
                fpm.add_reassociate_pass();
                fpm.add_gvn_pass();
                fpm.add_cfg_simplification_pass();
                fpm.add_basic_alias_analysis_pass();
                fpm.add_promote_memory_to_register_pass();
                fpm.add_instruction_combining_pass();
                fpm.add_reassociate_pass();
                fpm.initialize();

                let real_type = context.f64_type();
                let real_type_str = "f64";
                let ee = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();
                let mut codegen = CodeGen::new(model, &context, module, fpm, ee, real_type, real_type_str);

                let set_u0 = codegen.compile_set_u0(model)?;
                let residual = codegen.compile_residual(model)?;
                let calc_out = codegen.compile_calc_out(model)?;

                set_u0.print_to_stderr();
                residual.print_to_stderr();
                calc_out.print_to_stderr();

                Ok(codegen)
            }
        }.try_build()
    }

    pub fn jit_u0(&self) -> Result<JitFunction<U0Func>> {
        self.with_codegen(|codegen| {
            codegen.jit::<U0Func>("set_u0")
        })
    }

    pub fn jit_residual(&self) -> Result<JitFunction<ResidualFunc>> {
        self.with_codegen(|codegen| {
            codegen.jit::<ResidualFunc>("residual")
        })
    }

    pub fn jit_calc_out(&self) -> Result<JitFunction<CalcOutFunc>> {
        self.with_codegen(|codegen| {
            codegen.jit::<CalcOutFunc>("calc_out")
        })
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
        self.with_codegen(|codegen|
            target_machine.write_to_file(&codegen.module, FileType::Object, &path).map_err(|e| anyhow::anyhow!("Error writing object file: {:?}", e))
        )
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
        let object = ObjectCode::from_discrete_model(&discrete_model).unwrap();
        object.write_object_file().unwrap();

    }
}