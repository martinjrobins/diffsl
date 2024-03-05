use inkwell::basic_block::BasicBlock;
use inkwell::intrinsics::Intrinsic;
use inkwell::passes::PassManager;
use inkwell::types::{FloatType, BasicMetadataTypeEnum, BasicTypeEnum, IntType};
use inkwell::values::{PointerValue, FloatValue, FunctionValue, IntValue, BasicMetadataValueEnum, BasicValueEnum, BasicValue, GlobalValue};
use inkwell::{AddressSpace, IntPredicate};
use inkwell::builder::Builder;
use inkwell::module::{Module, Linkage};
use std::collections::HashMap;
use std::iter::zip;
use anyhow::{Result, anyhow};

type RealType = f64;


use crate::ast::{Ast, AstKind};
use crate::discretise::{DiscreteModel, Tensor, TensorBlock};
use crate::execution::{Translation, TranslationFrom, TranslationTo, DataLayout};

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
pub type StopFunc = unsafe extern "C" fn(time: RealType, u: *const RealType, up: *const RealType, data: *mut RealType, root: *mut RealType);
pub type ResidualFunc = unsafe extern "C" fn(time: RealType, u: *const RealType, up: *const RealType, data: *mut RealType, rr: *mut RealType);
pub type ResidualGradientFunc = unsafe extern "C" fn(time: RealType, u: *const RealType, du: *const RealType, up: *const RealType, dup: *const RealType, data: *mut RealType, ddata: *mut RealType, rr: *mut RealType, drr: *mut RealType);
pub type U0Func = unsafe extern "C" fn(data: *mut RealType, u: *mut RealType, up: *mut RealType);
pub type U0GradientFunc = unsafe extern "C" fn(data: *mut RealType, ddata: *mut RealType, u: *mut RealType, du: *mut RealType, up: *mut RealType, dup: *mut RealType);
pub type CalcOutFunc = unsafe extern "C" fn(time: RealType, u: *const RealType, up: *const RealType, data: *mut RealType);
pub type CalcOutGradientFunc = unsafe extern "C" fn(time: RealType, u: *const RealType, du: *const RealType, up: *const RealType, dup: *const RealType, data: *mut RealType, ddata: *mut RealType);
pub type GetDimsFunc = unsafe extern "C" fn(states: *mut u32, inputs: *mut u32, outputs: *mut u32, data: *mut u32, stop: *mut u32);
pub type SetInputsFunc = unsafe extern "C" fn(inputs: *const RealType, data: *mut RealType);
pub type SetInputsGradientFunc = unsafe extern "C" fn(inputs: *const RealType, dinputs: *const RealType, data: *mut RealType, ddata: *mut RealType);
pub type SetIdFunc = unsafe extern "C" fn(id: *mut RealType);
pub type GetOutFunc = unsafe extern "C" fn(data: *const RealType, tensor_data: *mut *mut RealType, tensor_size: *mut u32);

struct Globals<'ctx> {
    enzyme_dup: GlobalValue<'ctx>,
    enzyme_const: GlobalValue<'ctx>,
    enzyme_dupnoneed: GlobalValue<'ctx>,
    indices: GlobalValue<'ctx>,
}

impl<'ctx> Globals<'ctx> {
    fn new(layout: &DataLayout, context: &'ctx inkwell::context::Context, module: &Module<'ctx>) -> Result<Self> {
        let int_type = context.i32_type();
        let indices_array_type = int_type.array_type(u32::try_from(layout.indices().len()).unwrap());
        let indices_array_values = layout.indices().iter().map(|&i| int_type.const_int(i.try_into().unwrap(), false)).collect::<Vec<IntValue>>();
        let indices_value = int_type.const_array(indices_array_values.as_slice());
        let _int_ptr_type = int_type.ptr_type(AddressSpace::default());
        let globals = Self {
            enzyme_dup: module.add_global(int_type, Some(AddressSpace::default()), "enzyme_dup"),
            enzyme_const: module.add_global(int_type, Some(AddressSpace::default()), "enzyme_const"),
            enzyme_dupnoneed: module.add_global(int_type, Some(AddressSpace::default()), "enzyme_dupnoneed"),
            indices: module.add_global(indices_array_type, Some(AddressSpace::default()), "indices"),
        };
        globals.indices.set_initializer(&indices_value);
        Ok(globals)
    }
}

pub enum CompileGradientArgType {
    Const,
    Dup,
    DupNoNeed,
}


pub struct CodeGen<'ctx> {
    context: &'ctx inkwell::context::Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    fpm: PassManager<FunctionValue<'ctx>>,
    variables: HashMap<String, PointerValue<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    fn_value_opt: Option<FunctionValue<'ctx>>,
    tensor_ptr_opt: Option<PointerValue<'ctx>>,
    real_type: FloatType<'ctx>,
    real_type_str: String,
    int_type: IntType<'ctx>,
    layout: DataLayout,
    globals: Option<Globals<'ctx>>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(model: &DiscreteModel, context: &'ctx inkwell::context::Context, module: Module<'ctx>, real_type: FloatType<'ctx>, real_type_str: &str) -> Self {
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
        let builder = context.create_builder();
        let layout = DataLayout::new(model);
        let globals = Globals::new(&layout, context, &module).ok();
        Self {
            context,
            module,
            builder,
            fpm,
            real_type,
            real_type_str: real_type_str.to_owned(),
            variables: HashMap::new(),
            functions: HashMap::new(),
            fn_value_opt: None,
            tensor_ptr_opt: None,
            layout,
            int_type: context.i32_type(),
            globals,
        }
    }

    pub fn write_bitcode_to_path(&self, path: &std::path::Path) {
        self.module.write_bitcode_to_path(path);
    }

    fn insert_data(&mut self, model: &DiscreteModel) {
        for tensor in model.inputs() {
            self.insert_tensor(tensor);
        }
        for tensor in model.time_indep_defns() {
            self.insert_tensor(tensor);
        }
        for tensor in model.time_dep_defns() {
            self.insert_tensor(tensor);
        }
        for tensor in model.state_dep_defns() {
            self.insert_tensor(tensor);
        }
        self.insert_tensor(model.out());
        self.insert_tensor(model.lhs());
        self.insert_tensor(model.rhs());
    }
    
    fn insert_indices(&mut self) {
        let indices = self.globals.as_ref().unwrap().indices;
        let zero = self.context.i32_type().const_int(0, false);
        let ptr = unsafe { indices.as_pointer_value().const_in_bounds_gep(&[zero, zero]) };
        self.variables.insert("indices".to_owned(), ptr);
    }

    fn insert_param(&mut self, name: &str, value: PointerValue<'ctx>) {
        self.variables.insert(name.to_owned(), value);
    }

    fn insert_state(&mut self, u: &Tensor, dudt: &Tensor) {
        let mut data_index = 0;
        for blk in u.elmts() {
            if let Some(name) = blk.name() {
                let ptr = self.variables.get("u").unwrap();
                let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
                let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(*ptr, &[i], blk.name().unwrap()).unwrap() };
                self.variables.insert(name.to_owned(), alloca);
            }
            data_index += blk.nnz();
        }
        data_index = 0;
        for blk in dudt.elmts() {
            if let Some(name) = blk.name() {
                let ptr = self.variables.get("dudt").unwrap();
                let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
                let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(*ptr, &[i], blk.name().unwrap()).unwrap() };
                self.variables.insert(name.to_owned(), alloca);
            }
            data_index += blk.nnz();
        }
    }
    fn insert_tensor(&mut self, tensor: &Tensor) {
        let ptr = *self.variables.get("data").unwrap();
        let mut data_index = self.layout.get_data_index(tensor.name()).unwrap();
        let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
        let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(ptr, &[i], tensor.name()).unwrap() };
        self.variables.insert(tensor.name().to_owned(), alloca);
        
        //insert any named blocks
        for blk in tensor.elmts() {
            if let Some(name) = blk.name() {
                let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
                let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(ptr, &[i], name).unwrap() };
                self.variables.insert(name.to_owned(), alloca);
            }
            // named blocks only supported for rank <= 1, so we can just add the nnz to get the next data index
            data_index += blk.nnz();
        }
    }
    fn get_param(&self, name: &str) -> &PointerValue<'ctx> {
        self.variables.get(name).unwrap()
    }

    fn get_var(&self, tensor: &Tensor) -> &PointerValue<'ctx> {
        self.variables.get(tensor.name()).unwrap()
    }

    fn get_function(&mut self, name: &str) -> Option<FunctionValue<'ctx>> {
        match self.functions.get(name) {
            Some(&func) => Some(func),
            None => {
                let function = match name {
                    // support some llvm intrinsics
                    "sin" | "cos" | "tan" | "exp" | "log" | "log10" | "sqrt" | "abs" | "copysign" | "pow" | "min" | "max" => {
                        let arg_len = 1;
                        let intrinsic_name = match name {
                            "min" => "minnum",
                            "max" => "maxnum",
                            _ => name,
                        };
                        let llvm_name = format!("llvm.{}.{}", intrinsic_name, self.real_type_str);
                        let intrinsic = Intrinsic::find(&llvm_name).unwrap();
                        let ret_type = self.real_type;

                        let args_types = std::iter::repeat(ret_type)
                                    .take(arg_len)
                                    .map(|f| f.into())
                                    .collect::<Vec<BasicTypeEnum>>();
                        intrinsic.get_declaration(&self.module, args_types.as_slice())
                    },
                    // some custom functions
                    "sigmoid" => {
                        let arg_len = 1;
                        let ret_type = self.real_type;
                        
                        let args_types = std::iter::repeat(ret_type)
                            .take(arg_len)
                            .map(|f| f.into())
                            .collect::<Vec<BasicMetadataTypeEnum>>();
                        
                        let fn_type = ret_type.fn_type(args_types.as_slice(), false);
                        let fn_val = self.module.add_function(name, fn_type, None);
                        
                        for arg in fn_val.get_param_iter() {
                            arg.into_float_value().set_name("x");
                        }

                        let current_block = self.builder.get_insert_block().unwrap();
                        let basic_block = self.context.append_basic_block(fn_val, "entry");
                        self.builder.position_at_end(basic_block);
                        let x = fn_val.get_nth_param(0)?.into_float_value();
                        let one = self.real_type.const_float(1.0);
                        let negx = self.builder.build_float_neg(x, name).ok()?;
                        let exp = self.get_function("exp").unwrap();
                        let exp_negx = self.builder.build_call(exp, &[BasicMetadataValueEnum::FloatValue(negx)], name).ok()?;
                        let one_plus_exp_negx = self.builder.build_float_add(exp_negx.try_as_basic_value().left().unwrap().into_float_value(), one, name).ok()?;
                        let sigmoid = self.builder.build_float_div(one, one_plus_exp_negx, name).ok()?;
                        self.builder.build_return(Some(&sigmoid)).ok();
                        self.builder.position_at_end(current_block);
                        Some(fn_val)
                    },
                    "arcsinh" | "arccosh" => {
                        let arg_len = 1;
                        let ret_type = self.real_type;
                        
                        let args_types = std::iter::repeat(ret_type)
                            .take(arg_len)
                            .map(|f| f.into())
                            .collect::<Vec<BasicMetadataTypeEnum>>();
                        
                        let fn_type = ret_type.fn_type(args_types.as_slice(), false);
                        let fn_val = self.module.add_function(name, fn_type, None);
                        
                        for arg in fn_val.get_param_iter() {
                            arg.into_float_value().set_name("x");
                        }

                        let current_block = self.builder.get_insert_block().unwrap();
                        let basic_block = self.context.append_basic_block(fn_val, "entry");
                        self.builder.position_at_end(basic_block);
                        let x = fn_val.get_nth_param(0)?.into_float_value();
                        let one = match name {
                            "arccosh" => self.real_type.const_float(-1.0),
                            "arcsinh" => self.real_type.const_float(1.0),
                            _ => panic!("unknown function"),
                        };
                        let x_squared = self.builder.build_float_mul(x, x, name).ok()?;
                        let one_plus_x_squared = self.builder.build_float_add(x_squared, one, name).ok()?;
                        let sqrt = self.get_function("sqrt").unwrap();
                        let sqrt_one_plus_x_squared = self.builder.build_call(sqrt, &[BasicMetadataValueEnum::FloatValue(one_plus_x_squared)], name).unwrap()
                            .try_as_basic_value().left().unwrap().into_float_value();
                        let x_plus_sqrt_one_plus_x_squared = self.builder.build_float_add(x, sqrt_one_plus_x_squared, name).ok()?;
                        let ln = self.get_function("log").unwrap();
                        let result = self.builder.build_call(ln, &[BasicMetadataValueEnum::FloatValue(x_plus_sqrt_one_plus_x_squared)], name).unwrap()
                            .try_as_basic_value().left().unwrap().into_float_value();
                        self.builder.build_return(Some(&result)).ok();
                        self.builder.position_at_end(current_block);
                        Some(fn_val)
                    },
                    "tanh" | "sinh" | "cosh" => {
                        let arg_len = 1;
                        let ret_type = self.real_type;
                        
                        let args_types = std::iter::repeat(ret_type)
                            .take(arg_len)
                            .map(|f| f.into())
                            .collect::<Vec<BasicMetadataTypeEnum>>();
                        
                        let fn_type = ret_type.fn_type(args_types.as_slice(), false);
                        let fn_val = self.module.add_function(name, fn_type, None);
                        
                        for arg in fn_val.get_param_iter() {
                            arg.into_float_value().set_name("x");
                        }

                        let current_block = self.builder.get_insert_block().unwrap();
                        let basic_block = self.context.append_basic_block(fn_val, "entry");
                        self.builder.position_at_end(basic_block);
                        let x = fn_val.get_nth_param(0)?.into_float_value();
                        let negx = self.builder.build_float_neg(x, name).ok()?;
                        let exp = self.get_function("exp").unwrap();
                        let exp_negx = self.builder.build_call(exp, &[BasicMetadataValueEnum::FloatValue(negx)], name).ok()?;
                        let expx = self.builder.build_call(exp, &[BasicMetadataValueEnum::FloatValue(x)], name).ok()?;
                        let expx_minus_exp_negx = self.builder.build_float_sub(expx.try_as_basic_value().left().unwrap().into_float_value(), exp_negx.try_as_basic_value().left().unwrap().into_float_value(), name).ok()?;
                        let expx_plus_exp_negx = self.builder.build_float_add(expx.try_as_basic_value().left().unwrap().into_float_value(), exp_negx.try_as_basic_value().left().unwrap().into_float_value(), name).ok()?;
                        let result = match name {
                            "tanh" => {
                                self.builder.build_float_div(expx_minus_exp_negx, expx_plus_exp_negx, name).ok()?
                            },
                            "sinh" => {
                                self.builder.build_float_div(expx_minus_exp_negx, self.real_type.const_float(2.0), name).ok()?
                            },
                            "cosh" => {
                                self.builder.build_float_div(expx_plus_exp_negx, self.real_type.const_float(2.0), name).ok()?
                            },
                            _ => panic!("unknown function"),
                        };
                        self.builder.build_return(Some(&result)).ok();
                        self.builder.position_at_end(current_block);
                        Some(fn_val)
                    },
                    _ => None
                }?;
                self.functions.insert(name.to_owned(), function);
                Some(function)
            }
        }
    }
    /// Returns the `FunctionValue` representing the function being compiled.
    #[inline]
    fn fn_value(&self) -> FunctionValue<'ctx> {
        self.fn_value_opt.unwrap()
    }

    #[inline]
    fn tensor_ptr(&self) -> PointerValue<'ctx> {
        self.tensor_ptr_opt.unwrap()
    }

    /// Creates a new builder in the entry block of the function.
    fn create_entry_block_builder(&self) -> Builder<'ctx> {
        let builder = self.context.create_builder();
        let entry = self.fn_value().get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }
        builder
    }
    
    fn jit_compile_scalar(&mut self, a: &Tensor, res_ptr_opt: Option<PointerValue<'ctx>>)  -> Result<PointerValue<'ctx>> {
        let res_type = self.real_type;
        let res_ptr = match res_ptr_opt {
            Some(ptr) => ptr,
            None => self.create_entry_block_builder().build_alloca(res_type, a.name())?,
        };
        let name = a.name();
        let elmt = a.elmts().first().unwrap();
        let float_value = self.jit_compile_expr(name, elmt.expr(), &[], elmt, None)?;
        self.builder.build_store(res_ptr, float_value)?;
        Ok(res_ptr)
    }
    
    fn jit_compile_tensor(&mut self, a: &Tensor, res_ptr_opt: Option<PointerValue<'ctx>>)  -> Result<PointerValue<'ctx>> {
        // treat scalar as a special case
        if a.rank() == 0 {
            return self.jit_compile_scalar(a, res_ptr_opt)
        }

        let res_type = self.real_type;
        let res_ptr = match res_ptr_opt {
            Some(ptr) => ptr,
            None => self.create_entry_block_builder().build_alloca(res_type, a.name())?,
        };

        // set up the tensor storage pointer and index into this data
        self.tensor_ptr_opt = Some(res_ptr);

        for (i, blk) in a.elmts().iter().enumerate() {
            let default = format!("{}-{}", a.name(), i);
            let name = blk.name().unwrap_or(default.as_str());
            self.jit_compile_block(name, a, blk)?;
        }
        Ok(res_ptr)
    }

    fn jit_compile_block(&mut self, name: &str, tensor: &Tensor, elmt: &TensorBlock) -> Result<()> {
        let translation = Translation::new(elmt.expr_layout(), elmt.layout(), elmt.start(), tensor.layout_ptr());
        
        if elmt.expr_layout().is_dense() {
            self.jit_compile_dense_block(name, elmt, &translation)
        } else if elmt.expr_layout().is_diagonal() {
            self.jit_compile_diagonal_block(name, elmt, &translation)
        } else if elmt.expr_layout().is_sparse() {
            match translation.source {
                TranslationFrom::SparseContraction { .. } => {
                    self.jit_compile_sparse_contraction_block(name, elmt, &translation)
                },
                _ => {
                    self.jit_compile_sparse_block(name, elmt, &translation)
                }
            }
        } else {
            return Err(anyhow!("unsupported block layout: {:?}", elmt.expr_layout()));
        }

    }


    // for dense blocks we can loop through the nested loops to calculate the index, then we compile the expression passing in this index
    fn jit_compile_dense_block(&mut self, name: &str, elmt: &TensorBlock, translation: &Translation) -> Result<()> {
        let int_type = self.int_type;
        
        let mut preblock = self.builder.get_insert_block().unwrap();
        let expr_rank = elmt.expr_layout().rank();
        let expr_shape = elmt.expr_layout().shape().mapv(|n| int_type.const_int(n.try_into().unwrap(), false));
        let one = int_type.const_int(1, false);
        let zero = int_type.const_int(0, false);

        let expr_index_ptr = self.builder.build_alloca(int_type, "expr_index")?;
        let elmt_index_ptr = self.builder.build_alloca(int_type, "elmt_index")?;
        self.builder.build_store(expr_index_ptr, zero)?;
        self.builder.build_store(elmt_index_ptr, zero)?;

        // setup indices, loop through the nested loops
        let mut indices = Vec::new();
        let mut blocks = Vec::new();

        // allocate the contract sum if needed
        let (contract_sum, contract_by) = if let TranslationFrom::DenseContraction { contract_by, contract_len: _} = translation.source {
            (Some(self.builder.build_alloca(self.real_type, "contract_sum")?), contract_by)
        } else {
            (None, 0)
        };

        for i in 0..expr_rank {
            let block = self.context.append_basic_block(self.fn_value(), name);
            self.builder.build_unconditional_branch(block)?;
            self.builder.position_at_end(block);

            let start_index = int_type.const_int(0, false);
            let curr_index = self.builder.build_phi(int_type, format!["i{}", i].as_str())?;
            curr_index.add_incoming(&[(&start_index, preblock)]);

            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                self.builder.build_store(contract_sum.unwrap(), self.real_type.const_zero())?;
            }

            indices.push(curr_index);
            blocks.push(block);
            preblock = block;
        }


        let indices_int: Vec<IntValue> = indices.iter().map(|i| i.as_basic_value().into_int_value()).collect();

        // load and increment the expression index
        let expr_index = self.builder.build_load(expr_index_ptr, "expr_index")?.into_int_value();
        let elmt_index = self.builder.build_load(elmt_index_ptr, "elmt_index")?.into_int_value();
        let next_expr_index = self.builder.build_int_add(expr_index, one, "next_expr_index")?;
        self.builder.build_store(expr_index_ptr, next_expr_index)?;
        let float_value = self.jit_compile_expr(name, elmt.expr(), indices_int.as_slice(), elmt, Some(expr_index))?;

        if contract_sum.is_some() {
            let contract_sum_value = self.builder.build_load(contract_sum.unwrap(), "contract_sum")?.into_float_value();
            let new_contract_sum_value = self.builder.build_float_add(contract_sum_value, float_value, "new_contract_sum")?;
            self.builder.build_store(contract_sum.unwrap(), new_contract_sum_value)?;
        } else {
            preblock = self.jit_compile_broadcast_and_store(name, elmt, float_value, expr_index, translation, preblock)?;
            let next_elmt_index = self.builder.build_int_add(elmt_index, one, "next_elmt_index")?;
            self.builder.build_store(elmt_index_ptr, next_elmt_index)?;
        }
        
        // unwind the nested loops
        for i in (0..expr_rank).rev() {
            // increment index
            let next_index = self.builder.build_int_add(indices_int[i], one, name)?;
            indices[i].add_incoming(&[(&next_index, preblock)]);

            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                let contract_sum_value= self.builder.build_load(contract_sum.unwrap(), "contract_sum")?.into_float_value();
                let next_elmt_index = self.builder.build_int_add(elmt_index, one, "next_elmt_index")?;
                self.builder.build_store(elmt_index_ptr, next_elmt_index)?;
                self.jit_compile_store(name, elmt, elmt_index, contract_sum_value, translation)?;
            }

            // loop condition
            let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, expr_shape[i], name)?;
            let block = self.context.append_basic_block(self.fn_value(), name);
            self.builder.build_conditional_branch(loop_while, blocks[i], block)?;
            self.builder.position_at_end(block);
            preblock = block;
        }
        Ok(())
    }


    fn jit_compile_sparse_contraction_block(&mut self, name: &str, elmt: &TensorBlock, translation: &Translation) -> Result<()> {
        match translation.source {
            TranslationFrom::SparseContraction {..} => {},
            _ => {
                panic!("expected sparse contraction")
            }
        }
        let int_type = self.int_type;
        
        let preblock = self.builder.get_insert_block().unwrap();
        let layout_index = self.layout.get_layout_index(elmt.expr_layout()).unwrap();
        let translation_index = self.layout.get_translation_index(elmt.expr_layout(), elmt.layout()).unwrap();
        let translation_index = translation_index + translation.get_from_index_in_data_layout();

        let contract_sum_ptr = self.builder.build_alloca(self.real_type, "contract_sum")?;


        // loop through each contraction 
        let block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(block)?;
        self.builder.position_at_end(block);

        let contract_index = self.builder.build_phi(int_type, "i")?;
        let final_contract_index = int_type.const_int(elmt.layout().nnz().try_into().unwrap(), false);
        contract_index.add_incoming(&[(&int_type.const_int(0, false), preblock)]);

        let start_index = self.builder.build_int_add(
            int_type.const_int(translation_index.try_into().unwrap(), false),
            self.builder.build_int_mul(
                int_type.const_int(2, false),
                contract_index.as_basic_value().into_int_value(),
                name,
            )?,
            name
        )?;
        let end_index = self.builder.build_int_add(
            start_index,
            int_type.const_int(1, false),
            name
        )?;
        let start_ptr = unsafe { self.builder.build_gep(*self.get_param("indices"), &[start_index], "start_index_ptr")? };
        let start_contract= self.builder.build_load(start_ptr, "start")?.into_int_value();
        let end_ptr = unsafe { self.builder.build_gep(*self.get_param("indices"), &[end_index], "end_index_ptr")? };
        let end_contract = self.builder.build_load(end_ptr, "end")?.into_int_value();

        // initialise the contract sum
        self.builder.build_store(contract_sum_ptr, self.real_type.const_float(0.0))?;

        // loop through each element in the contraction
        let contract_block = self.context.append_basic_block(self.fn_value(), format!("{}_contract", name).as_str());
        self.builder.build_unconditional_branch(contract_block)?;
        self.builder.position_at_end(contract_block);

        let expr_index_phi = self.builder.build_phi(int_type, "j")?;
        expr_index_phi.add_incoming(&[(&start_contract, block)]);

        // loop body - load index from layout
        let expr_index = expr_index_phi.as_basic_value().into_int_value();
        let elmt_index_mult_rank = self.builder.build_int_mul(expr_index, int_type.const_int(elmt.expr_layout().rank().try_into().unwrap(), false), name)?;
        let indices_int = (0..elmt.expr_layout().rank()).map(|i| {
            let layout_index_plus_offset = int_type.const_int((layout_index + i).try_into().unwrap(), false);
            let curr_index = self.builder.build_int_add(elmt_index_mult_rank, layout_index_plus_offset, name)?;
            let ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("indices"), &[curr_index], name)? };
            let index = self.builder.build_load(ptr, name)?.into_int_value();
            Ok(index)
        }).collect::<Result<Vec<_>, anyhow::Error>>()?;
        
        // loop body - eval expression and increment sum
        let float_value = self.jit_compile_expr(name, elmt.expr(), indices_int.as_slice(), elmt, Some(expr_index))?;
        let contract_sum_value = self.builder.build_load(contract_sum_ptr, "contract_sum")?.into_float_value();
        let new_contract_sum_value = self.builder.build_float_add(contract_sum_value, float_value, "new_contract_sum")?;
        self.builder.build_store(contract_sum_ptr, new_contract_sum_value)?;

        // increment contract loop index
        let next_elmt_index = self.builder.build_int_add(expr_index, int_type.const_int(1, false), name)?;
        expr_index_phi.add_incoming(&[(&next_elmt_index, contract_block)]);

        // contract loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_elmt_index, end_contract, name)?;
        let post_contract_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, contract_block, post_contract_block)?;
        self.builder.position_at_end(post_contract_block);

        // store the result
        self.jit_compile_store(name, elmt, contract_index.as_basic_value().into_int_value(), new_contract_sum_value, translation)?;

        // increment outer loop index
        let next_contract_index = self.builder.build_int_add(contract_index.as_basic_value().into_int_value(), int_type.const_int(1, false), name)?;
        contract_index.add_incoming(&[(&next_contract_index, post_contract_block)]);

        // outer loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_contract_index, final_contract_index, name)?;
        let post_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, block, post_block)?;
        self.builder.position_at_end(post_block);

        Ok(())
    }
    
    // for sparse blocks we can loop through the non-zero elements and extract the index from the layout, then we compile the expression passing in this index
    // TODO: havn't implemented contractions yet
    fn jit_compile_sparse_block(&mut self, name: &str, elmt: &TensorBlock, translation: &Translation) -> Result<()> {

        let int_type = self.int_type;
        
        let preblock = self.builder.get_insert_block().unwrap();
        let layout_index = self.layout.get_layout_index(elmt.expr_layout()).unwrap();
        // loop through the non-zero elements
        let mut block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(block)?;
        self.builder.position_at_end(block);

        let start_index = int_type.const_int(0, false);
        let end_index = int_type.const_int(elmt.expr_layout().nnz().try_into().unwrap(), false);
        let curr_index = self.builder.build_phi(int_type, "i")?;
        curr_index.add_incoming(&[(&start_index, preblock)]);
        
        // loop body - load index from layout
        let elmt_index = curr_index.as_basic_value().into_int_value();
        let elmt_index_mult_rank = self.builder.build_int_mul(elmt_index, int_type.const_int(elmt.expr_layout().rank().try_into().unwrap(), false), name)?;
        let indices_int = (0..elmt.expr_layout().rank()).map(|i| {
            let layout_index_plus_offset = int_type.const_int((layout_index + i).try_into().unwrap(), false);
            let curr_index = self.builder.build_int_add(elmt_index_mult_rank, layout_index_plus_offset, name)?;
            let ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("indices"), &[curr_index], name)? };
            Ok(self.builder.build_load(ptr, name)?.into_int_value())
        }).collect::<Result<Vec<_>, anyhow::Error>>()?;
        
        // loop body - eval expression
        let float_value = self.jit_compile_expr(name, elmt.expr(), indices_int.as_slice(), elmt, Some(elmt_index))?;

        block = self.jit_compile_broadcast_and_store(name, elmt, float_value, elmt_index, translation, block)?;

        // increment loop index
        let one = int_type.const_int(1, false);
        let next_index = self.builder.build_int_add(elmt_index, one, name)?;
        curr_index.add_incoming(&[(&next_index, block)]);

        // loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, end_index, name)?;
        let post_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, block, post_block)?;
        self.builder.position_at_end(post_block);

        Ok(())
    }
    
    // for diagonal blocks we can loop through the diagonal elements and the index is just the same for each element, then we compile the expression passing in this index
    fn jit_compile_diagonal_block(&mut self, name: &str, elmt: &TensorBlock, translation: &Translation) -> Result<()> {
        let int_type = self.int_type;
        
        let preblock = self.builder.get_insert_block().unwrap();

        // loop through the non-zero elements
        let mut block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(block)?;
        self.builder.position_at_end(block);

        let start_index = int_type.const_int(0, false);
        let end_index = int_type.const_int(elmt.expr_layout().nnz().try_into().unwrap(), false);
        let curr_index = self.builder.build_phi(int_type, "i")?;
        curr_index.add_incoming(&[(&start_index, preblock)]);
        
        // loop body - index is just the same for each element
        let elmt_index = curr_index.as_basic_value().into_int_value();
        let indices_int: Vec<IntValue> = (0..elmt.expr_layout().rank()).map(|_| {
            elmt_index
        }).collect();
        
        // loop body - eval expression
        let float_value = self.jit_compile_expr(name, elmt.expr(), indices_int.as_slice(), elmt, Some(elmt_index))?;

        // loop body - store result
        block = self.jit_compile_broadcast_and_store(name, elmt, float_value, elmt_index, translation, block)?;
        
        // increment loop index
        let one = int_type.const_int(1, false);
        let next_index = self.builder.build_int_add(elmt_index, one, name)?;
        curr_index.add_incoming(&[(&next_index, block)]);

        // loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, end_index, name)?;
        let post_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, block, post_block)?;
        self.builder.position_at_end(post_block);

        Ok(())
    }

    fn jit_compile_broadcast_and_store(&mut self, name: &str, elmt: &TensorBlock, float_value: FloatValue<'ctx>, expr_index: IntValue<'ctx>, translation: &Translation, pre_block: BasicBlock<'ctx>) -> Result<BasicBlock<'ctx>> {
        let int_type = self.int_type;
        let one = int_type.const_int(1, false);
        let zero = int_type.const_int(0, false);
        match translation.source {
            TranslationFrom::Broadcast { broadcast_by: _, broadcast_len } => {
                let bcast_start_index = zero;
                let bcast_end_index = int_type.const_int(broadcast_len.try_into().unwrap(), false);

                // setup loop block
                let bcast_block = self.context.append_basic_block(self.fn_value(), name);
                self.builder.build_unconditional_branch(bcast_block)?;
                self.builder.position_at_end(bcast_block);
                let bcast_index = self.builder.build_phi(int_type, "broadcast_index")?;
                bcast_index.add_incoming(&[(&bcast_start_index, pre_block)]);

                // store value
                let store_index = self.builder.build_int_add(
                    self.builder.build_int_mul(expr_index, bcast_end_index, "store_index")?,
                    bcast_index.as_basic_value().into_int_value(),
                    "bcast_store_index"
                )?;
                self.jit_compile_store(name, elmt, store_index, float_value, translation)?;

                // increment index
                let bcast_next_index= self.builder.build_int_add(bcast_index.as_basic_value().into_int_value(), one, name)?;
                bcast_index.add_incoming(&[(&bcast_next_index, bcast_block)]);

                // loop condition
                let bcast_cond = self.builder.build_int_compare(IntPredicate::ULT, bcast_next_index, bcast_end_index, "broadcast_cond")?;
                let post_bcast_block = self.context.append_basic_block(self.fn_value(), name);
                self.builder.build_conditional_branch(bcast_cond, bcast_block, post_bcast_block)?;
                self.builder.position_at_end(post_bcast_block);

                // return the current block for later
                Ok(post_bcast_block)

            },
            TranslationFrom::ElementWise | TranslationFrom::DiagonalContraction { .. } => {
                self.jit_compile_store(name, elmt, expr_index, float_value, translation)?;
                Ok(pre_block)
            },
            _ => Err(anyhow!("Invalid translation")),
        }
    }


    fn jit_compile_store(&mut self, name: &str, elmt: &TensorBlock, store_index: IntValue<'ctx>, float_value: FloatValue<'ctx>, translation: &Translation) -> Result<()> {
        let int_type = self.int_type;
        let rank = elmt.layout().rank();
        let res_index = match &translation.target {
            TranslationTo::Contiguous { start, end: _ } => {
                let start_const = int_type.const_int((*start).try_into().unwrap(), false);
                self.builder.build_int_add(start_const, store_index, name)?
            },
            TranslationTo::Sparse { indices: _ } => {
                // load store index from layout
                let translate_index = self.layout.get_translation_index(elmt.expr_layout(), elmt.layout()).unwrap();
                let translate_store_index = translate_index + translation.get_to_index_in_data_layout();
                let translate_store_index = int_type.const_int(translate_store_index.try_into().unwrap(), false);
                let rank_const = int_type.const_int(rank.try_into().unwrap(), false);
                let elmt_index_strided = self.builder.build_int_mul(store_index, rank_const, name)?;
                let curr_index = self.builder.build_int_add(elmt_index_strided, translate_store_index, name)?;
                let ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("indices"), &[curr_index], name)? };
                self.builder.build_load(ptr, name)?.into_int_value()
            },
        };
        let resi_ptr = unsafe { self.builder.build_in_bounds_gep(self.tensor_ptr(), &[res_index], name)? };
        self.builder.build_store(resi_ptr, float_value)?;
        Ok(())
    }

    fn jit_compile_expr(&mut self, name: &str, expr: &Ast, index: &[IntValue<'ctx>], elmt: &TensorBlock, expr_index: Option<IntValue<'ctx>>) -> Result<FloatValue<'ctx>> {
        let name= elmt.name().unwrap_or(name);
        match &expr.kind {
            AstKind::Binop(binop) => {
                let lhs = self.jit_compile_expr(name, binop.left.as_ref(), index, elmt, expr_index)?;
                let rhs = self.jit_compile_expr(name, binop.right.as_ref(), index, elmt, expr_index)?;
                match binop.op {
                    '*' => Ok(self.builder.build_float_mul(lhs, rhs, name)?),
                    '/' => Ok(self.builder.build_float_div(lhs, rhs, name)?),
                    '-' => Ok(self.builder.build_float_sub(lhs, rhs, name)?),
                    '+' => Ok(self.builder.build_float_add(lhs, rhs, name)?),
                    unknown => Err(anyhow!("unknown binop op '{}'", unknown))
                }
            },
            AstKind::Monop(monop) => {
                let child = self.jit_compile_expr(name, monop.child.as_ref(), index, elmt, expr_index)?;
                match monop.op {
                    '-' => Ok(self.builder.build_float_neg(child, name)?),
                    unknown => Err(anyhow!("unknown monop op '{}'", unknown))
                }                
            },
            AstKind::Call(call) => {
                match self.get_function(call.fn_name) {
                    Some(function) => {
                        let mut args: Vec<BasicMetadataValueEnum> = Vec::new();
                        for arg in call.args.iter() {
                            let arg_val = self.jit_compile_expr(name, arg.as_ref(), index, elmt, expr_index)?;
                            args.push(BasicMetadataValueEnum::FloatValue(arg_val));
                        }
                        let ret_value = self.builder.build_call(function, args.as_slice(), name)?
                            .try_as_basic_value().left().unwrap().into_float_value();
                        Ok(ret_value)
                    },
                    None => Err(anyhow!("unknown function call '{}'", call.fn_name))
                }
                
            },
            AstKind::CallArg(arg) => {
                self.jit_compile_expr(name, &arg.expression, index, elmt, expr_index)
            },
            AstKind::Number(value) => Ok(self.real_type.const_float(*value)),
            AstKind::IndexedName(iname) => {
                let ptr = self.get_param(iname.name);
                let layout = self.layout.get_layout(iname.name).unwrap();
                let iname_elmt_index = if layout.is_dense() {
                    // permute indices based on the index chars of this tensor
                    let mut no_transform = true;
                    let mut iname_index = Vec::new();
                    for (i, c) in iname.indices.iter().enumerate() {
                        // find the position index of this index char in the tensor's index chars,
                        // if it's not found then it must be a contraction index so is at the end
                        let pi = elmt.indices().iter().position(|x| x == c).unwrap_or(elmt.indices().len());
                        iname_index.push(index[pi]);
                        no_transform = no_transform && pi == i;
                    }
                    // calculate the element index using iname_index and the shape of the tensor
                    // TODO: can we optimise this by using expr_index, and also including elmt_index?
                    if !iname_index.is_empty() {
                        let mut iname_elmt_index = *iname_index.last().unwrap();
                        let mut stride = 1u64;
                        for i in (0..iname_index.len() - 1).rev() {
                            let iname_i = iname_index[i];
                            let shapei: u64 = layout.shape()[i + 1].try_into().unwrap();
                            stride *= shapei;
                            let stride_intval = self.context.i32_type().const_int(stride, false);
                            let stride_mul_i = self.builder.build_int_mul(stride_intval, iname_i, name)?;
                            iname_elmt_index = self.builder.build_int_add(iname_elmt_index, stride_mul_i, name)?;
                        }
                        Some(iname_elmt_index)
                    } else {
                        let zero = self.context.i32_type().const_int(0, false);
                        Some(zero)

                    }
                } else if layout.is_sparse() || layout.is_diagonal() {
                    // must have come from jit_compile_sparse_block, so we can just use the elmt_index
                    // must have come from jit_compile_diagonal_block, so we can just use the elmt_index
                    expr_index
                } else {
                    panic!("unexpected layout");
                };
                let value_ptr = match iname_elmt_index {
                    Some(index) => unsafe { self.builder.build_in_bounds_gep(*ptr, &[index], name)? },
                    None => *ptr
                };
                Ok(self.builder.build_load(value_ptr, name)?.into_float_value())
            },
            AstKind::Name(name) => {
                // must be a scalar, just load the value
                let ptr = self.get_param(name);
                Ok(self.builder.build_load(*ptr, name)?.into_float_value())
            },
            AstKind::NamedGradient(name) => {
                let name_str = name.to_string();
                let ptr = self.get_param(name_str.as_str());
                Ok(self.builder.build_load(*ptr, name_str.as_str())?.into_float_value())
            },
            AstKind::Index(_) => todo!(),
            AstKind::Slice(_) => todo!(),
            AstKind::Integer(_) => todo!(),
            _ => panic!("unexprected astkind"),
        }
    }

    
    fn clear(&mut self) {
        self.variables.clear();
        self.functions.clear();
        self.fn_value_opt = None;
        self.tensor_ptr_opt = None;
    }
    
    fn function_arg_alloca(&mut self, name: &str, arg: BasicValueEnum<'ctx>) -> PointerValue<'ctx> {
        match arg {
            BasicValueEnum::PointerValue(v) => v,
            BasicValueEnum::FloatValue(v) => {
                    let alloca = self.create_entry_block_builder().build_alloca(arg.get_type(), name).unwrap();
                    self.builder.build_store(alloca, v).unwrap();
                    alloca
                }
            _ => unreachable!()
        }
    }

    

    pub fn compile_set_u0<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &[ "data", "u0", "dudt0"];
        let function = self.module.add_function("set_u0", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);
        self.builder.position_at_end(basic_block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }

        self.insert_data(model);
        self.insert_indices();

        for a in model.time_indep_defns() {
            self.jit_compile_tensor(a, Some(*self.get_var(a)))?;
        }

        self.jit_compile_tensor(model.state(), Some(*self.get_param("u0")))?;
        self.jit_compile_tensor(model.state_dot(), Some(*self.get_param("dudt0")))?;

        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);

            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }

    pub fn compile_calc_out<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[self.real_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &["t", "u", "dudt", "data"];
        let function = self.module.add_function("calc_out", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);
        self.builder.position_at_end(basic_block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }

        self.insert_state(model.state(), model.state_dot());
        self.insert_data(model);
        self.insert_indices();
        
        // TODO: could split state dep defns into before and after F and G

        self.jit_compile_tensor(model.out(), Some(*self.get_var(model.out())))?;
        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);

            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }

    pub fn compile_calc_stop<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[self.real_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &["t", "u", "dudt", "data", "root"];
        let function = self.module.add_function("calc_stop", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);
        self.builder.position_at_end(basic_block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }

        self.insert_state(model.state(), model.state_dot());
        self.insert_data(model);
        self.insert_indices();
        
        if let Some(stop) = model.stop() {
            let res_ptr = self.get_param("root");
            self.jit_compile_tensor(stop, Some(*res_ptr))?;
        }
        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);

            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }


    pub fn compile_residual<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[self.real_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &["t", "u", "dudt", "data", "rr"];
        let function = self.module.add_function("residual", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);
        self.builder.position_at_end(basic_block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }

        self.insert_state(model.state(), model.state_dot());
        self.insert_data(model);
        self.insert_indices();

        // calculate time dependant definitions
        for tensor in model.time_dep_defns() {
            self.jit_compile_tensor(tensor, Some(*self.get_var(tensor)))?;
        }
        
        // TODO: could split state dep defns into before and after F and G
        for a in model.state_dep_defns() {
            self.jit_compile_tensor(a, Some(*self.get_var(a)))?;
        }

        // F and G
        self.jit_compile_tensor(model.lhs(), Some(*self.get_var(model.lhs())))?;
        self.jit_compile_tensor(model.rhs(), Some(*self.get_var(model.rhs())))?;
        
        // compute residual here as dummy array
        let residual = model.residual();

        let res_ptr = self.get_param("rr");
        let _res_ptr = self.jit_compile_tensor(&residual, Some(*res_ptr))?;
        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);
            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }
    
    pub fn compile_gradient(&mut self, original_function: FunctionValue<'ctx>, args_type: &[CompileGradientArgType]) -> Result<FunctionValue<'ctx>> {
        self.clear();

        let globals = match self.globals {
            Some(ref globals) => globals,
            None => panic!("globals not set"),
        };
        
        // construct the gradient function
        let mut fn_type: Vec<BasicMetadataTypeEnum> = Vec::new();
        let orig_fn_type_ptr = original_function.get_type().ptr_type(AddressSpace::default());
        let mut enzyme_fn_type: Vec<BasicMetadataTypeEnum> = vec![orig_fn_type_ptr.into()];
        let mut start_param_index: Vec<u32> = Vec::new();
        for (i, arg) in original_function.get_param_iter().enumerate() {
            start_param_index.push(u32::try_from(fn_type.len()).unwrap());
            fn_type.push(arg.get_type().into());

            // constant args with type T in the original funciton have 2 args of type [int, T]
            enzyme_fn_type.push(self.int_type.into());
            enzyme_fn_type.push(arg.get_type().into());

            match args_type[i] {
                CompileGradientArgType::Dup | CompileGradientArgType::DupNoNeed => {
                    fn_type.push(arg.get_type().into());
                    enzyme_fn_type.push(arg.get_type().into());
                }
                CompileGradientArgType::Const => {},
            }
        }
        let void_type = self.context.void_type(); 
        let fn_type = void_type.fn_type(fn_type.as_slice(), false);
        let fn_name = format!("{}_grad", original_function.get_name().to_str().unwrap());
        let function = self.module.add_function(fn_name.as_str(), fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);
        self.builder.position_at_end(basic_block);

        let mut enzyme_fn_args: Vec<BasicMetadataValueEnum> = vec![original_function.as_global_value().as_pointer_value().into()];
        let enzyme_const = self.builder.build_load(globals.enzyme_const.as_pointer_value(), "enzyme_const")?;
        let enzyme_dup= self.builder.build_load(globals.enzyme_dup.as_pointer_value(), "enzyme_dup")?;
        let enzyme_dupnoneed = self.builder.build_load(globals.enzyme_dupnoneed.as_pointer_value(), "enzyme_dupnoneed")?;
        for (i, _arg) in original_function.get_param_iter().enumerate() {
            let param_index = start_param_index[i];
            let fn_arg = function.get_nth_param(param_index).unwrap();
            match args_type[i] {
                CompileGradientArgType::Dup => {
                    // let enzyme know its an active arg
                    enzyme_fn_args.push(enzyme_dup.into());

                    // pass in the arg value
                    enzyme_fn_args.push(fn_arg.into());

                    // pass in the darg value
                    let fn_darg = function.get_nth_param(param_index + 1).unwrap();
                    enzyme_fn_args.push(fn_darg.into());
                },
                CompileGradientArgType::DupNoNeed => {
                    // let enzyme know its an active arg we don't need
                    enzyme_fn_args.push(enzyme_dupnoneed.into());

                    // pass in the arg value
                    enzyme_fn_args.push(fn_arg.into());

                    // pass in the darg value
                    let fn_darg = function.get_nth_param(param_index + 1).unwrap();
                    enzyme_fn_args.push(fn_darg.into());
                },
                CompileGradientArgType::Const => {
                    // let enzyme know its a constant arg
                    enzyme_fn_args.push(enzyme_const.into());

                    // pass in the arg value
                    enzyme_fn_args.push(fn_arg.into());

                },
            }
        }       
        
        // construct enzyme function
        // double df = __enzyme_fwddiff<double>((void*)f, enzyme_dup, x, dx, enzyme_dup, y, dy); 
        let enzyme_fn_type = void_type.fn_type(&enzyme_fn_type, false);
        let orig_fn_name = original_function.get_name().to_str().unwrap();
        let enzyme_fn_name = format!("__enzyme_fwddiff_{}", orig_fn_name);
        let enzyme_function = self.module.add_function(enzyme_fn_name.as_str(), enzyme_fn_type, Some(Linkage::External));
        
        // call enzyme function
        self.builder.build_call(enzyme_function, enzyme_fn_args.as_slice(), "enzyme_call")?;
        
        // return
        self.builder.build_return(None)?;
        
        if function.verify(true) {
            self.fpm.run_on(&function);
            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }

    pub fn compile_get_dims(&mut self, model: &DiscreteModel) ->  Result<FunctionValue<'ctx>> {
        self.clear();
        let int_ptr_type = self.context.i32_type().ptr_type(AddressSpace::default());
        let fn_type = self.context.void_type().fn_type(
            &[int_ptr_type.into(), int_ptr_type.into(), int_ptr_type.into(), int_ptr_type.into(), int_ptr_type.into()]
            , false
        );

        let function = self.module.add_function("get_dims", fn_type, None);
        let block = self.context.append_basic_block(function, "entry");
        let fn_arg_names = &["states", "inputs", "outputs", "data", "stop"];
        self.builder.position_at_end(block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }
        
        self.insert_indices();

        let number_of_states = model.state().nnz() as u64;
        let number_of_inputs = model.inputs().iter().fold(0, |acc, x| acc + x.nnz()) as u64;
        let number_of_outputs = model.out().nnz() as u64;
        let number_of_stop = if let Some(stop) = model.stop() {
            stop.nnz() as u64
        } else {
            0
        };
        let data_len = self.layout.data().len() as u64;
        self.builder.build_store(*self.get_param("states"), self.int_type.const_int(number_of_states, false))?;
        self.builder.build_store(*self.get_param("inputs"), self.int_type.const_int(number_of_inputs, false))?;
        self.builder.build_store(*self.get_param("outputs"), self.int_type.const_int(number_of_outputs, false))?;
        self.builder.build_store(*self.get_param("data"), self.int_type.const_int(data_len, false))?;
        self.builder.build_store(*self.get_param("stop"), self.int_type.const_int(number_of_stop, false))?;
        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);
            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }

    pub fn compile_get_tensor(&mut self, model: &DiscreteModel, name: &str) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_ptr_type = self.real_type.ptr_type(AddressSpace::default()).ptr_type(AddressSpace::default());
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let int_ptr_type = self.context.i32_type().ptr_type(AddressSpace::default());
        let fn_type = self.context.void_type().fn_type(
            &[real_ptr_type.into(), real_ptr_ptr_type.into(), int_ptr_type.into()]
            , false
        );
        let function_name = format!("get_{}", name);
        let function = self.module.add_function(function_name.as_str(), fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);

        let fn_arg_names = &["data", "tensor_data", "tensor_size"];
        self.builder.position_at_end(basic_block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }

        self.insert_data(model);
        let ptr = self.get_param(name);
        let tensor_size = self.layout.get_layout(name).unwrap().nnz() as u64;
        let tensor_size_value =  self.int_type.const_int(tensor_size, false);
        self.builder.build_store(*self.get_param("tensor_data"), ptr.as_basic_value_enum())?;
        self.builder.build_store(*self.get_param("tensor_size"), tensor_size_value)?;
        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);
            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }

    pub fn compile_set_inputs(&mut self, model: &DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let function = self.module.add_function("set_inputs", fn_type, None);
        let mut block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);

        let fn_arg_names = &["inputs", "data"];
        self.builder.position_at_end(block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }

        let mut inputs_index = 0usize;
        for input in model.inputs() {
            let name = format!("input_{}", input.name());
            self.insert_tensor(input);
            let ptr = self.get_var(input);
            // loop thru the elements of this input and set them using the inputs ptr
            let inputs_start_index = self.int_type.const_int(inputs_index as u64, false);
            let start_index = self.int_type.const_int(0, false);
            let end_index = self.int_type.const_int(input.nnz().try_into().unwrap(), false);

            let input_block = self.context.append_basic_block(function, name.as_str());
            self.builder.build_unconditional_branch(input_block)?;
            self.builder.position_at_end(input_block);
            let index = self.builder.build_phi(self.int_type, "i")?;
            index.add_incoming(&[(&start_index, block)]);
            
            // loop body - copy value from inputs to data
            let curr_input_index = index.as_basic_value().into_int_value();
            let input_ptr = unsafe { self.builder.build_in_bounds_gep(*ptr, &[curr_input_index], name.as_str())? };
            let curr_inputs_index = self.builder.build_int_add(inputs_start_index, curr_input_index, name.as_str())?;
            let inputs_ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("inputs"), &[curr_inputs_index], name.as_str())? };
            let input_value = self.builder.build_load(inputs_ptr, name.as_str())?.into_float_value();
            self.builder.build_store(input_ptr, input_value)?;

            // increment loop index
            let one = self.int_type.const_int(1, false);
            let next_index = self.builder.build_int_add(curr_input_index, one, name.as_str())?;
            index.add_incoming(&[(&next_index, input_block)]);

            // loop condition
            let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, end_index, name.as_str())?;
            let post_block = self.context.append_basic_block(function, name.as_str());
            self.builder.build_conditional_branch(loop_while, input_block, post_block)?;
            self.builder.position_at_end(post_block);

            // get ready for next input
            block = post_block;
            inputs_index += input.nnz();
        }
        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);
            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }

    pub fn compile_set_id(&mut self, model: &DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[real_ptr_type.into()]
            , false
        );
        let function = self.module.add_function("set_id", fn_type, None);
        let mut block = self.context.append_basic_block(function, "entry");

        let fn_arg_names = &["id"];
        self.builder.position_at_end(block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.insert_param(name, alloca);
        }

        let mut id_index = 0usize;
        for (blk, is_algebraic) in zip(model.state().elmts(), model.is_algebraic()) {
            let name = blk.name().unwrap_or("unknown");
            // loop thru the elements of this state blk and set the corresponding elements of id
            let id_start_index = self.int_type.const_int(id_index as u64, false);
            let blk_start_index = self.int_type.const_int(0, false);
            let blk_end_index = self.int_type.const_int(blk.nnz().try_into().unwrap(), false);

            let blk_block = self.context.append_basic_block(function, name);
            self.builder.build_unconditional_branch(blk_block)?;
            self.builder.position_at_end(blk_block);
            let index = self.builder.build_phi(self.int_type, "i")?;
            index.add_incoming(&[(&blk_start_index, block)]);
            
            // loop body - copy value from inputs to data
            let curr_blk_index = index.as_basic_value().into_int_value();
            let curr_id_index = self.builder.build_int_add(id_start_index, curr_blk_index, name)?;
            let id_ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("id"), &[curr_id_index], name)? };
            let is_algebraic_float = if *is_algebraic { 0.0 as RealType } else { 1.0 as RealType };
            let is_algebraic_value = self.real_type.const_float(is_algebraic_float);
            self.builder.build_store(id_ptr, is_algebraic_value)?;

            // increment loop index
            let one = self.int_type.const_int(1, false);
            let next_index = self.builder.build_int_add(curr_blk_index, one, name)?;
            index.add_incoming(&[(&next_index, blk_block)]);

            // loop condition
            let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, blk_end_index, name)?;
            let post_block = self.context.append_basic_block(function, name);
            self.builder.build_conditional_branch(loop_while, blk_block, post_block)?;
            self.builder.position_at_end(post_block);

            // get ready for next blk
            block = post_block;
            id_index += blk.nnz();
        }
        self.builder.build_return(None)?;

        if function.verify(true) {
            self.fpm.run_on(&function);
            Ok(function)
        } else {
            function.print_to_stderr();
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }



    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }
}