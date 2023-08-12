use inkwell::basic_block::BasicBlock;
use inkwell::intrinsics::Intrinsic;
use inkwell::passes::PassManager;
use inkwell::types::{FloatType, BasicMetadataTypeEnum, BasicTypeEnum};
use inkwell::values::{PointerValue, FloatValue, FunctionValue, IntValue, BasicMetadataValueEnum, BasicValueEnum};
use inkwell::{OptimizationLevel, AddressSpace, IntPredicate};
use inkwell::builder::Builder;
use inkwell::execution_engine::{ExecutionEngine, JitFunction, UnsafeFunctionPointer};
use inkwell::module::Module;
use ndarray::{Array1, Array2, ShapeBuilder, s};
use sundials_sys::{realtype, N_Vector, IDAGetNonlinSolvStats, IDA_SUCCESS, IDA_ROOT_RETURN, IDA_YA_YDP_INIT, IDA_NORMAL, IDASolve, IDAGetIntegratorStats, IDASetStopTime, IDACreate, N_VNew_Serial, N_VGetArrayPointer, N_VConst, IDAInit, IDACalcIC, IDASVtolerances, IDASetUserData, SUNLinSolInitialize, IDASetId, SUNMatrix, SUNLinearSolver, SUNDenseMatrix, PREC_NONE, PREC_LEFT, SUNLinSol_Dense, SUNLinSol_SPBCGS, SUNLinSol_SPFGMR, SUNLinSol_SPGMR, SUNLinSol_SPTFQMR, IDASetLinearSolver, SUNLinSolFree, SUNMatDestroy, N_VDestroy, IDAFree, IDAReInit, IDAGetConsistentIC, IDAGetReturnFlagName};
use std::collections::HashMap;
use std::io::Write;
use std::io;
use std::ffi::{c_void, CStr, c_int};
use std::ptr::null_mut;
use std::iter::zip;
use anyhow::{Result, anyhow};


use crate::ast::{Ast, AstKind};
use crate::discretise::{DiscreteModel, Tensor, DataLayout, TensorBlock, Translation, TranslationFrom, TranslationTo, Layout};

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type ResidualFunc = unsafe extern "C" fn(time: realtype, u: *const realtype, up: *const realtype, data: *mut realtype, indices: *const i32, rr: *mut realtype);
type U0Func = unsafe extern "C" fn(data: *mut realtype, indices: *const i32, u: *mut realtype, up: *mut realtype);
type CalcOutFunc = unsafe extern "C" fn(time: realtype, u: *const realtype, up: *const realtype, data: *mut realtype, indices: *const i32);

struct CodeGen<'ctx> {
    context: &'ctx inkwell::context::Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    fpm: PassManager<FunctionValue<'ctx>>,
    ee: ExecutionEngine<'ctx>,
    variables: HashMap<String, PointerValue<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    fn_value_opt: Option<FunctionValue<'ctx>>,
    tensor_ptr_opt: Option<PointerValue<'ctx>>,
    real_type: FloatType<'ctx>,
    real_type_str: String,
    layout: DataLayout,
}

impl<'ctx> CodeGen<'ctx> {
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
    fn insert_param(&mut self, name: &str, value: PointerValue<'ctx>) {
        self.variables.insert(name.to_owned(), value);
    }
    fn insert_state(&mut self, u: &Tensor, dudt: &Tensor) {
        let mut data_index = 0;
        for blk in u.elmts() {
            if let Some(name) = blk.name() {
                let ptr = self.variables.get("u").unwrap();
                let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
                let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(*ptr, &[i], blk.name().unwrap()) };
                self.variables.insert(name.to_owned(), alloca);
            }
            data_index += blk.nnz();
        }
        data_index = 0;
        for blk in dudt.elmts() {
            if let Some(name) = blk.name() {
                let ptr = self.variables.get("dudt").unwrap();
                let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
                let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(*ptr, &[i], blk.name().unwrap()) };
                self.variables.insert(name.to_owned(), alloca);
            }
            data_index += blk.nnz();
        }
    }
    fn insert_tensor(&mut self, tensor: &Tensor) {
        let ptr = self.variables.get("data").unwrap().clone();
        let mut data_index = self.layout.get_data_index(tensor.name()).unwrap();
        let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
        let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(ptr, &[i], tensor.name()) };
        self.variables.insert(tensor.name().to_owned(), alloca);
        
        //insert any named blocks
        for blk in tensor.elmts() {
            if let Some(name) = blk.name() {
                let i = self.context.i32_type().const_int(data_index.try_into().unwrap(), false);
                let alloca = unsafe { self.create_entry_block_builder().build_in_bounds_gep(ptr, &[i], name) };
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
            // support some llvm intrinsics
            None => {
                match name {
                    "sin" | "cos" | "tan" | "pow" | "exp" | "log" | "sqrt" | "abs" => {
                        let arg_len = 1;
                        let llvm_name = format!("llvm.{}.{}", name, self.real_type_str);
                        let intrinsic = Intrinsic::find(&llvm_name).unwrap();
                        let ret_type = self.real_type;
                        let args_types = std::iter::repeat(ret_type)
                            .take(arg_len)
                            .map(|f| f.into())
                            .collect::<Vec<BasicMetadataTypeEnum>>();
                        let args_types = args_types.as_slice();
                        let fn_type = ret_type.fn_type(args_types, false);
                        let fn_val = self.module.add_function(name, fn_type, None);

                        for (_, arg) in fn_val.get_param_iter().enumerate() {
                            arg.into_float_value().set_name("x");
                        }

                        let args_types = std::iter::repeat(ret_type)
                            .take(arg_len)
                            .map(|f| f.into())
                            .collect::<Vec<BasicTypeEnum>>();
                        let args_types = args_types.as_slice();
                        let function = intrinsic.get_declaration(&self.module, args_types).unwrap();

                        self.functions.insert(name.to_owned(), function)
                    },
                    _ => None,
                }

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
            None => self.create_entry_block_builder().build_alloca(res_type, a.name()),
        };
        let name = a.name();
        let elmt = a.elmts().first().unwrap();
        let float_value = self.jit_compile_expr(name, &elmt.expr(), &[], elmt, None)?;
        self.builder.build_store(res_ptr, float_value);
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
            None => self.create_entry_block_builder().build_alloca(res_type, a.name()),
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
        let int_type = self.context.i64_type();
        
        let mut preblock = self.builder.get_insert_block().unwrap();
        let expr_rank = elmt.expr_layout().rank();
        let expr_shape = elmt.expr_layout().shape().mapv(|n| int_type.const_int(n.try_into().unwrap(), false));
        let one = int_type.const_int(1, false);
        let zero = int_type.const_int(0, false);

        let expr_index_ptr = self.builder.build_alloca(int_type, "expr_index");
        let elmt_index_ptr = self.builder.build_alloca(int_type, "elmt_index");
        self.builder.build_store(expr_index_ptr, zero);
        self.builder.build_store(elmt_index_ptr, zero);

        // setup indices, loop through the nested loops
        let mut indices = Vec::new();
        let mut blocks = Vec::new();


        // allocate the contract sum if needed
        let (contract_sum, contract_by) = if let TranslationFrom::DenseContraction { contract_by, contract_len: _} = translation.source {
            (Some(self.builder.build_alloca(self.real_type, "contract_sum")), contract_by)
        } else {
            (None, 0)
        };

        for i in 0..expr_rank {
            let block = self.context.append_basic_block(self.fn_value(), name);
            self.builder.build_unconditional_branch(block);
            self.builder.position_at_end(block);

            let start_index = int_type.const_int(0, false);
            let curr_index = self.builder.build_phi(int_type, format!["i{}", i].as_str());
            curr_index.add_incoming(&[(&start_index, preblock)]);

            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                self.builder.build_store(contract_sum.unwrap(), self.real_type.const_zero());
            }

            indices.push(curr_index);
            blocks.push(block);
            preblock = block;
        }


        let indices_int: Vec<IntValue> = indices.iter().map(|i| i.as_basic_value().into_int_value()).collect();

        // load and increment the expression index
        let expr_index = self.builder.build_load(expr_index_ptr, "expr_index").into_int_value();
        let elmt_index = self.builder.build_load(elmt_index_ptr, "elmt_index").into_int_value();
        let next_expr_index = self.builder.build_int_add(expr_index, one, "next_expr_index");
        self.builder.build_store(expr_index_ptr, next_expr_index);
        let float_value = self.jit_compile_expr(name, &elmt.expr(), indices_int.as_slice(), elmt, Some(expr_index))?;

        if contract_sum.is_some() {
            let contract_sum_value = self.builder.build_load(contract_sum.unwrap(), "contract_sum").into_float_value();
            let new_contract_sum_value = self.builder.build_float_add(contract_sum_value, float_value, "new_contract_sum");
            self.builder.build_store(contract_sum.unwrap(), new_contract_sum_value);
        } else {
            preblock = self.jit_compile_broadcast_and_store(name, elmt, float_value, expr_index, translation, preblock)?;
            let next_elmt_index = self.builder.build_int_add(elmt_index, one, "next_elmt_index");
            self.builder.build_store(elmt_index_ptr, next_elmt_index);
        }
        
        // unwind the nested loops
        for i in (0..expr_rank).rev() {
            // increment index
            let next_index = self.builder.build_int_add(indices_int[i], one, name);
            indices[i].add_incoming(&[(&next_index, preblock)]);

            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                let contract_sum_value= self.builder.build_load(contract_sum.unwrap(), "contract_sum").into_float_value();
                let next_elmt_index = self.builder.build_int_add(elmt_index, one, "next_elmt_index");
                self.builder.build_store(elmt_index_ptr, next_elmt_index);
                self.jit_compile_store(name, elmt, elmt_index, contract_sum_value, translation)?;
            }

            // loop condition
            let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, expr_shape[i], name);
            let block = self.context.append_basic_block(self.fn_value(), name);
            self.builder.build_conditional_branch(loop_while, blocks[i], block);
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
        let int_type = self.context.i64_type();
        
        let preblock = self.builder.get_insert_block().unwrap();
        let layout_index = self.layout.get_layout_index(elmt.layout()).unwrap();
        let translation_index = self.layout.get_translation_index(elmt.expr_layout(), elmt.layout()).unwrap();
        let translation_index = translation_index + translation.get_from_index_in_data_layout();

        let contract_sum_ptr = self.builder.build_alloca(self.real_type, "contract_sum");


        // loop through each contraction 
        let block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(block);
        self.builder.position_at_end(block);

        let contract_index = self.builder.build_phi(int_type, "i");
        let final_contract_index = int_type.const_int(elmt.layout().nnz().try_into().unwrap(), false);
        contract_index.add_incoming(&[(&int_type.const_int(0, false), preblock)]);

        let start_index = self.builder.build_int_add(
            int_type.const_int(translation_index.try_into().unwrap(), false),
            self.builder.build_int_mul(
                int_type.const_int(2, false),
                contract_index.as_basic_value().into_int_value(),
                name,
            ),
            name
        );
        let end_index = self.builder.build_int_add(
            start_index,
            int_type.const_int(1, false),
            name
        );
        let start_ptr = unsafe { self.builder.build_gep(*self.get_param("indices"), &[start_index], "start_index_ptr") };
        let start_contract= self.builder.build_load(start_ptr, "start").into_int_value();
        let end_ptr = unsafe { self.builder.build_gep(*self.get_param("indices"), &[end_index], "end_index_ptr") };
        let end_contract = self.builder.build_load(end_ptr, "end").into_int_value();

        // loop through each element in the contraction
        let contract_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(contract_block);
        self.builder.position_at_end(contract_block);

        self.builder.build_store(contract_sum_ptr, self.real_type.const_float(0.0));

        let elmt_index_phi = self.builder.build_phi(int_type, "j");
        elmt_index_phi.add_incoming(&[(&start_contract, block)]);

        // loop body - load index from layout
        let elmt_index = elmt_index_phi.as_basic_value().into_int_value();
        let indices_int: Vec<IntValue> = (0..elmt.expr_layout().rank()).map(|i| {
            let layout_index_plus_offset = int_type.const_int((layout_index + i).try_into().unwrap(), false);
            let curr_index = self.builder.build_int_add(elmt_index, layout_index_plus_offset, name);
            let ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("indices"), &[curr_index], name) };
            self.builder.build_load(ptr, name).into_int_value()
        }).collect();
        
        // loop body - eval expression and increment sum
        let float_value = self.jit_compile_expr(name, &elmt.expr(), indices_int.as_slice(), elmt, Some(elmt_index))?;
        let contract_sum_value = self.builder.build_load(contract_sum_ptr, "contract_sum").into_float_value();
        let contract_sum_value = self.builder.build_float_add(contract_sum_value, float_value, "contract_sum");
        self.builder.build_store(contract_sum_ptr, contract_sum_value);

        // increment contract loop index
        let next_elmt_index = self.builder.build_int_add(elmt_index, int_type.const_int(1, false), name);
        elmt_index_phi.add_incoming(&[(&next_elmt_index, contract_block)]);

        // contract loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_elmt_index, end_contract, name);
        let post_contract_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, contract_block, post_contract_block);
        self.builder.position_at_end(post_contract_block);

        // increment outer loop index
        let next_contract_index = self.builder.build_int_add(contract_index.as_basic_value().into_int_value(), int_type.const_int(1, false), name);
        contract_index.add_incoming(&[(&next_contract_index, post_contract_block)]);

        // outer loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_contract_index, final_contract_index, name);
        let post_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, block, post_block);

        Ok(())
    }
    
    // for sparse blocks we can loop through the non-zero elements and extract the index from the layout, then we compile the expression passing in this index
    // TODO: havn't implemented contractions yet
    fn jit_compile_sparse_block(&mut self, name: &str, elmt: &TensorBlock, translation: &Translation) -> Result<()> {

        let int_type = self.context.i64_type();
        
        let preblock = self.builder.get_insert_block().unwrap();
        let layout_index = self.layout.get_layout_index(elmt.expr_layout()).unwrap();

        // loop through the non-zero elements
        let mut block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(block);
        self.builder.position_at_end(block);

        let start_index = int_type.const_int(0, false);
        let end_index = int_type.const_int(elmt.expr_layout().nnz().try_into().unwrap(), false);
        let curr_index = self.builder.build_phi(int_type, "i");
        curr_index.add_incoming(&[(&start_index, preblock)]);
        
        // loop body - load index from layout
        let elmt_index = curr_index.as_basic_value().into_int_value();
        let indices_int: Vec<IntValue> = (0..elmt.expr_layout().rank()).map(|i| {
            let layout_index_plus_offset = int_type.const_int((layout_index + i).try_into().unwrap(), false);
            let curr_index = self.builder.build_int_add(elmt_index, layout_index_plus_offset, name);
            let ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("indices"), &[curr_index], name) };
            self.builder.build_load(ptr, name).into_int_value()
        }).collect();
        
        // loop body - eval expression
        let float_value = self.jit_compile_expr(name, &elmt.expr(), indices_int.as_slice(), elmt, Some(elmt_index))?;

        block = self.jit_compile_broadcast_and_store(name, elmt, float_value, elmt_index, translation, block)?;

        // increment loop index
        let one = int_type.const_int(1, false);
        let next_index = self.builder.build_int_add(elmt_index, one, name);
        curr_index.add_incoming(&[(&next_index, block)]);

        // loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, end_index, name);
        let post_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, block, post_block);
        self.builder.position_at_end(post_block);

        Ok(())
    }
    
    // for diagonal blocks we can loop through the diagonal elements and the index is just the same for each element, then we compile the expression passing in this index
    fn jit_compile_diagonal_block(&mut self, name: &str, elmt: &TensorBlock, translation: &Translation) -> Result<()> {
        let int_type = self.context.i64_type();
        
        let preblock = self.builder.get_insert_block().unwrap();

        // loop through the non-zero elements
        let mut block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(block);
        self.builder.position_at_end(block);

        let start_index = int_type.const_int(0, false);
        let end_index = int_type.const_int(elmt.expr_layout().nnz().try_into().unwrap(), false);
        let curr_index = self.builder.build_phi(int_type, "i");
        curr_index.add_incoming(&[(&start_index, preblock)]);
        
        // loop body - index is just the same for each element
        let elmt_index = curr_index.as_basic_value().into_int_value();
        let indices_int: Vec<IntValue> = (0..elmt.expr_layout().rank()).map(|_| {
            elmt_index.clone()
        }).collect();
        
        // loop body - eval expression
        let float_value = self.jit_compile_expr(name, &elmt.expr(), indices_int.as_slice(), elmt, Some(elmt_index))?;

        // loop body - store result
        block = self.jit_compile_broadcast_and_store(name, elmt, float_value, elmt_index, translation, block)?;
        
        // increment loop index
        let one = int_type.const_int(1, false);
        let next_index = self.builder.build_int_add(elmt_index, one, name);
        curr_index.add_incoming(&[(&next_index, block)]);

        // loop condition
        let loop_while = self.builder.build_int_compare(IntPredicate::ULT, next_index, end_index, name);
        let post_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_conditional_branch(loop_while, block, post_block);
        self.builder.position_at_end(post_block);

        Ok(())
    }

    fn jit_compile_broadcast_and_store(&mut self, name: &str, elmt: &TensorBlock, float_value: FloatValue<'ctx>, expr_index: IntValue<'ctx>, translation: &Translation, pre_block: BasicBlock<'ctx>) -> Result<BasicBlock<'ctx>> {
        let int_type = self.context.i64_type();
        let one = int_type.const_int(1, false);
        let zero = int_type.const_int(0, false);
        match translation.source {
            TranslationFrom::Broadcast { broadcast_by: _, broadcast_len } => {
                let bcast_start_index = zero;
                let bcast_end_index = int_type.const_int(broadcast_len.try_into().unwrap(), false);

                // setup loop block
                let bcast_block = self.context.append_basic_block(self.fn_value(), name);
                self.builder.build_unconditional_branch(bcast_block);
                self.builder.position_at_end(bcast_block);
                let bcast_index = self.builder.build_phi(int_type, "broadcast_index");
                bcast_index.add_incoming(&[(&bcast_start_index, pre_block)]);

                // store value
                let store_index = self.builder.build_int_add(
                    self.builder.build_int_mul(expr_index, bcast_end_index, "store_index"),
                    bcast_index.as_basic_value().into_int_value(),
                    "bcast_store_index"
                );
                self.jit_compile_store(name, elmt, store_index, float_value, translation)?;

                // increment index
                let bcast_next_index= self.builder.build_int_add(bcast_index.as_basic_value().into_int_value(), one, name);
                bcast_index.add_incoming(&[(&bcast_next_index, bcast_block)]);

                // loop condition
                let bcast_cond = self.builder.build_int_compare(IntPredicate::ULT, bcast_next_index, bcast_end_index, "broadcast_cond");
                let post_bcast_block = self.context.append_basic_block(self.fn_value(), name);
                self.builder.build_conditional_branch(bcast_cond, bcast_block, post_bcast_block);
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
        let int_type = self.context.i64_type();
        let rank = elmt.layout().rank();
        let res_index = match &translation.target {
            TranslationTo::Contiguous { start, end: _ } => {
                let start_const = int_type.const_int((*start).try_into().unwrap(), false);
                self.builder.build_int_add(start_const, store_index, name)
            },
            TranslationTo::Sparse { indices: _ } => {
                // load store index from layout
                let translate_index = self.layout.get_translation_index(elmt.expr_layout(), elmt.layout()).unwrap();
                let translate_store_index = translate_index + translation.get_to_index_in_data_layout();
                let translate_store_index = int_type.const_int(translate_store_index.try_into().unwrap(), false);
                let rank_const = int_type.const_int(rank.try_into().unwrap(), false);
                let elmt_index_strided = self.builder.build_int_mul(store_index, rank_const, name);
                let curr_index = self.builder.build_int_add(elmt_index_strided, translate_store_index, name);
                let ptr = unsafe { self.builder.build_in_bounds_gep(*self.get_param("indices"), &[curr_index], name) };
                self.builder.build_load(ptr, name).into_int_value()
            },
        };
        let resi_ptr = unsafe { self.builder.build_in_bounds_gep(self.tensor_ptr(), &[res_index], name) };
        self.builder.build_store(resi_ptr, float_value);
        Ok(())
    }

    fn jit_compile_expr(&mut self, name: &str, expr: &Ast, index: &[IntValue<'ctx>], elmt: &TensorBlock, expr_index: Option<IntValue<'ctx>>) -> Result<FloatValue<'ctx>> {
        let name= elmt.name().unwrap_or(name);
        match &expr.kind {
            AstKind::Binop(binop) => {
                let lhs = self.jit_compile_expr(name, binop.left.as_ref(), index, elmt, expr_index)?;
                let rhs = self.jit_compile_expr(name, binop.right.as_ref(), index, elmt, expr_index)?;
                match binop.op {
                    '*' => Ok(self.builder.build_float_mul(lhs, rhs, name)),
                    '/' => Ok(self.builder.build_float_div(lhs, rhs, name)),
                    '-' => Ok(self.builder.build_float_sub(lhs, rhs, name)),
                    '+' => Ok(self.builder.build_float_add(lhs, rhs, name)),
                    unknown => Err(anyhow!("unknown binop op '{}'", unknown))
                }
            },
            AstKind::Monop(monop) => {
                let child = self.jit_compile_expr(name, monop.child.as_ref(), index, elmt, expr_index)?;
                match monop.op {
                    '-' => Ok(self.builder.build_float_neg(child, name)),
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
                        let ret_value = self.builder.build_call(function, args.as_slice(), name)
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
                    if iname_index.len() > 0 {
                        let mut iname_elmt_index = iname_index.last().unwrap().clone();
                        let mut stride = 1u64;
                        for i in (0..iname_index.len() - 1).rev() {
                            let iname_i = iname_index[i];
                            let shapei: u64 = layout.shape()[i + 1].try_into().unwrap();
                            stride *= shapei;
                            let stride_intval = self.context.i64_type().const_int(stride, false);
                            let stride_mul_i = self.builder.build_int_mul(stride_intval, iname_i, name);
                            iname_elmt_index = self.builder.build_int_add(iname_elmt_index, stride_mul_i, name);
                        }
                        Some(iname_elmt_index)
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        Some(zero)

                    }
                } else if layout.is_sparse() {
                    // must have come from jit_compile_sparse_block, so we can just use the elmt_index
                    expr_index
                } else if layout.is_diagonal() {
                    // must have come from jit_compile_diagonal_block, so we can just use the elmt_index
                    expr_index
                } else {
                    panic!("unexpected layout");
                };
                let value_ptr = match iname_elmt_index {
                    Some(index) => unsafe { self.builder.build_in_bounds_gep(*ptr, &[index], name) },
                    None => *ptr
                };
                Ok(self.builder.build_load(value_ptr, name).into_float_value())
            },
            AstKind::Name(name) => {
                // must be a scalar, just load the value
                let ptr = self.get_param(name);
                Ok(self.builder.build_load(*ptr, name).into_float_value())
            },
            AstKind::NamedGradient(name) => {
                let name_str = name.to_string();
                let ptr = self.get_param(name_str.as_str());
                Ok(self.builder.build_load(*ptr, name_str.as_str()).into_float_value())
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
                    let alloca = self.create_entry_block_builder().build_alloca(arg.get_type(), name);
                    self.builder.build_store(alloca, v);
                    alloca
                }
            _ => unreachable!()
        }
    }

    fn jit<T>(&self, function: FunctionValue) -> Result<JitFunction<'ctx, T>> 
    where T: UnsafeFunctionPointer
    {
        let name = function.get_name().to_str().unwrap();
        let maybe_fn = unsafe { self.ee.get_function::<T>(name) };
        let compiled_fn = match maybe_fn {
            Ok(f) => Ok(f),
            Err(err) => {
                Err(anyhow!("Error during jit for {}: {}", name, err))
            },
        };
        compiled_fn
    }

    fn compile_set_u0<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let int_ptr_type = self.context.i32_type().ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[real_ptr_type.into(), int_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &[ "data", "indices", "u0", "dudt0"];
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

        for a in model.time_indep_defns() {
            self.jit_compile_tensor(a, Some(*self.get_var(a)))?;
        }

        self.jit_compile_tensor(&model.state(), Some(*self.get_param("u0")))?;
        self.jit_compile_tensor(&model.state_dot(), Some(*self.get_param("dudt0")))?;

        self.builder.build_return(None);

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

    fn compile_calc_out<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let int_ptr_type = self.context.i32_type().ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[self.real_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into(), int_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &["t", "u", "dudt", "data", "indices", "out"];
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

        self.jit_compile_tensor(model.out(), Some(*self.get_var(model.out())))?;
        self.builder.build_return(None);

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

    fn compile_residual<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let int_ptr_type = self.context.i32_type().ptr_type(AddressSpace::default());
        let fn_type = void_type.fn_type(
            &[self.real_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into(), int_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &["t", "u", "dudt", "data", "indices", "rr"];
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

        // calculate time dependant definitions
        for tensor in model.time_dep_defns() {
            self.jit_compile_tensor(tensor, Some(*self.get_var(tensor)))?;
        }

        // F and G
        self.jit_compile_tensor(&model.lhs(), Some(*self.get_var(model.lhs())))?;
        self.jit_compile_tensor(&model.rhs(), Some(*self.get_var(model.rhs())))?;
        
        // compute residual here as dummy array
        let residual = model.residual();

        let res_ptr = self.get_param("rr");
        let _res_ptr = self.jit_compile_tensor(&residual, Some(*res_ptr))?;
        self.builder.build_return(None);

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
}

pub struct Options {
    atol: f64,
    rtol: f64,
    print_stats: bool,
    jacobian: String,
    linear_solver: String, // klu, lapack, spbcg 
    preconditioner: String, // spbcg 
    linsol_max_iterations: u32,
}

impl Options {
    pub fn new() -> Options {
        Options {
            atol: 1e-6,
            rtol: 1e-6,
            print_stats: false,
            jacobian: "none".to_owned(),
            linear_solver: "SUNLinSol_Dense".to_owned(),
            preconditioner: "none".to_owned(),
            linsol_max_iterations: 10,
        }
    }
}


struct SundialsData<'ctx> {
    number_of_states: usize,
    number_of_parameters: usize,
    input_names: Vec<String>,
    yy: N_Vector,
    yp: N_Vector, 
    avtol: N_Vector,
    data: N_Vector,
    data_layout: DataLayout,
    yy_s: Vec<N_Vector>, 
    yp_s: Vec<N_Vector>,
    id: N_Vector,
    jacobian: SUNMatrix,
    linear_solver: SUNLinearSolver, 
    residual: JitFunction<'ctx, ResidualFunc>,
    set_u0: JitFunction<'ctx, U0Func>,
    calc_out: JitFunction<'ctx, CalcOutFunc>,
    options: Options,
}

impl<'ctx> SundialsData<'ctx> {
    pub fn get_tensor_data(&self, name: &str) -> Option<&[realtype]> {
        let index = self.data_layout.get_data_index(name)?;
        let nnz = self.data_layout.get_data_length(name)?;
        unsafe {
            Some(std::slice::from_raw_parts(
                N_VGetArrayPointer(self.data).offset(index.try_into().unwrap()),
                nnz
            ))
        }
    }
    pub fn get_tensor_data_mut(&self, name: &str) -> Option<&mut [realtype]> {
        let index = self.data_layout.get_data_index(name)?;
        let nnz = self.data_layout.get_data_length(name)?;
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                N_VGetArrayPointer(self.data).offset(index.try_into().unwrap()),
                nnz
            ))
        }
    }
    pub fn get_data_ptr_mut(&self) -> *mut realtype {
        unsafe {
            N_VGetArrayPointer(self.data)
        }
    }
}

pub struct Sundials<'ctx> {
    ida_mem: *mut c_void, // pointer to memory
    data: Box<SundialsData<'ctx>>,
}

impl<'ctx> Sundials<'ctx> {
    unsafe extern "C" fn sresidual(
        t: realtype,
        y: N_Vector,
        yp: N_Vector,
        rr: N_Vector,
        user_data: *mut c_void,
    ) -> i32 {
        let data = & *(user_data as *mut SundialsData);

        data.residual.call(t, 
            N_VGetArrayPointer(y), 
            N_VGetArrayPointer(yp), 
            N_VGetArrayPointer(data.data), 
            data.data_layout.indices().as_ptr(),
            N_VGetArrayPointer(rr), 
        );
        io::stdout().flush().unwrap();
        0
    }   

    fn check(retval: c_int) -> Result<()> {
        if retval < 0 {
            let char_ptr = unsafe { IDAGetReturnFlagName(i64::from(retval)) };
            let c_str = unsafe { CStr::from_ptr(char_ptr) };
            Err(anyhow!("Sundials Error Name: {}", c_str.to_str()?))
        } else {
            return Ok(())
        }
    }

    fn set_inputs(&mut self, inputs: &Array1<f64>) -> Result<()> {
        let number_of_inputs = inputs.len();
        assert_eq!(number_of_inputs, self.data.number_of_parameters);
        let mut curr_index = 0;
        for name in self.data.input_names.iter() {
            let data = self.data.get_tensor_data_mut(name).unwrap();
            data.copy_from_slice(&inputs.slice(s![curr_index..curr_index + data.len()]).as_slice().unwrap());
            curr_index += data.len();
        }
        Ok(())
    }

    pub fn calc_u0(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        self.set_inputs(inputs).unwrap();
        let mut u0 = Array1::zeros(self.data.number_of_states);
        unsafe {
            let data_ptr = N_VGetArrayPointer(self.data.data);
            let indices_ptr = self.data.data_layout.indices().as_ptr();
            self.data.set_u0.call(data_ptr, indices_ptr, N_VGetArrayPointer(self.data.yy), N_VGetArrayPointer(self.data.yp));
            let yy_ptr = N_VGetArrayPointer(self.data.yy);
            for i in 0..self.data.number_of_states {
                u0[i] = *yy_ptr.add(i); 
            }
        }
        u0
    }

    pub fn calc_residual(&mut self, t: f64, inputs: &Array1<f64>, u0: &Array1<f64>, up0: &Array1<f64>) -> Array1<f64> {
        self.set_inputs(inputs).unwrap();
        let number_of_states = u0.len();
        assert_eq!(number_of_states, up0.len());
        assert_eq!(number_of_states, self.data.number_of_states);
        let mut res = Array1::zeros(number_of_states);
        unsafe {
            let rr = N_VNew_Serial(i64::try_from(number_of_states).unwrap());
            let u0_ptr = N_VGetArrayPointer(self.data.yy);
            let up0_ptr = N_VGetArrayPointer(self.data.yp);
            for i in 0..number_of_states {
                *u0_ptr.add(i) = u0[i]; 
                *up0_ptr.add(i) = up0[i]; 
            }
            self.data.residual.call(t, 
                N_VGetArrayPointer(self.data.yy), 
                N_VGetArrayPointer(self.data.yp), 
                self.data.get_data_ptr_mut(),
                self.data.data_layout.indices().as_ptr(),
                N_VGetArrayPointer(rr), 
            );
            io::stdout().flush().unwrap();
            let rr_ptr = N_VGetArrayPointer(rr);
            for i in 0..self.data.number_of_states {
                res[i] = *rr_ptr.add(i); 
            }
        }
        res
    }

    pub fn calc_out(&mut self, t: f64, inputs: &Array1<f64>, u0: &Array1<f64>, up0: &Array1<f64>) -> Array1<f64> {
        self.set_inputs(inputs).unwrap();
        let name = "out";
        let number_of_states = u0.len();
        assert_eq!(number_of_states, up0.len());
        assert_eq!(number_of_states, self.data.number_of_states);
        let tensor_nnz = self.data.data_layout.get_data_length(name).unwrap();
        let mut res = Array1::zeros(tensor_nnz);
        unsafe {
            let u0_ptr = N_VGetArrayPointer(self.data.yy);
            let up0_ptr = N_VGetArrayPointer(self.data.yp);
            for i in 0..number_of_states {
                *u0_ptr.add(i) = u0[i]; 
                *up0_ptr.add(i) = up0[i]; 
            }

            self.data.calc_out.call(t, 
                N_VGetArrayPointer(self.data.yy), 
                N_VGetArrayPointer(self.data.yp), 
                self.data.get_data_ptr_mut(),
                self.data.data_layout.indices().as_ptr(),
            );
            io::stdout().flush().unwrap();
            let f = self.data.get_tensor_data(name).unwrap();
            for i in 0..tensor_nnz {
                res[i] = f[i]; 
            }
        }
        res
    }

    pub fn calc_tensor(&mut self, name: &str, t: f64, inputs: &Array1<f64>, u0: &Array1<f64>, up0: &Array1<f64>) -> Array1<f64> {
        self.set_inputs(inputs).unwrap();
        let number_of_states = u0.len();
        assert_eq!(number_of_states, up0.len());
        assert_eq!(number_of_states, self.data.number_of_states);
        let tensor_nnz = self.data.data_layout.get_data_length(name).unwrap();
        let mut res = Array1::zeros(tensor_nnz);
        unsafe {
            let rr = N_VNew_Serial(i64::try_from(number_of_states).unwrap());
            let u0_ptr = N_VGetArrayPointer(self.data.yy);
            let up0_ptr = N_VGetArrayPointer(self.data.yp);
            for i in 0..number_of_states {
                *u0_ptr.add(i) = u0[i]; 
                *up0_ptr.add(i) = up0[i]; 
            }
            self.data.residual.call(t, 
                N_VGetArrayPointer(self.data.yy), 
                N_VGetArrayPointer(self.data.yp), 
                self.data.get_data_ptr_mut(),
                self.data.data_layout.indices().as_ptr(),
                N_VGetArrayPointer(rr), 
            );
            io::stdout().flush().unwrap();
            let f = self.data.get_tensor_data(name).unwrap();
            for i in 0..tensor_nnz {
                res[i] = f[i]; 
            }
        }
        res
    }

    pub fn from_discrete_model<'m>(model: &'m DiscreteModel, context: &'ctx inkwell::context::Context, options: Options) -> Result<Sundials<'ctx>> {
        let number_of_states = i64::try_from(
            *model.state().shape().first().unwrap_or(&1)
        ).unwrap();
        let number_of_parameters = model.inputs().iter().fold(0, |acc, input| acc + i64::try_from(input.nnz()).unwrap());
        let input_names = model.inputs().iter().map(|input| input.name().to_owned()).collect::<Vec<_>>();
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
        let ee = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();
        let real_type = context.f64_type();
        let real_type_str = "f64";
        let mut codegen = CodeGen {
            context: &context,
            module,
            builder: context.create_builder(),
            fpm,
            ee,
            real_type,
            real_type_str: real_type_str.to_owned(),
            variables: HashMap::new(),
            functions: HashMap::new(),
            fn_value_opt: None,
            tensor_ptr_opt: None,
            layout: DataLayout::new(model),
        };

        let set_u0 = codegen.compile_set_u0(model)?;
        let residual = codegen.compile_residual(model)?;
        let calc_out = codegen.compile_calc_out(model)?;

        set_u0.print_to_stderr();
        residual.print_to_stderr();
        calc_out.print_to_stderr();

        let set_u0 = codegen.jit::<U0Func>(set_u0)?;
        let residual = codegen.jit::<ResidualFunc>(residual)?;
        let calc_out = codegen.jit::<CalcOutFunc>(calc_out)?;

        let data_layout = model.create_data_layout();
        
        unsafe {
            let ida_mem = IDACreate();

            // allocate vectors
            let yy = N_VNew_Serial(number_of_states);
            let yp = N_VNew_Serial(number_of_states);
            let avtol = N_VNew_Serial(i64::from(number_of_states));
            let id = N_VNew_Serial(i64::from(number_of_states));

            let data = N_VNew_Serial(data_layout.data().len().try_into().unwrap());

            let mut yy_s: Vec<N_Vector> = Vec::new();
            let mut yp_s: Vec<N_Vector> = Vec::new();
            for _ in 0..number_of_parameters {
                yy_s.push(N_VNew_Serial(i64::from(number_of_parameters)));
                yp_s.push(N_VNew_Serial(i64::from(number_of_parameters)));
            }

            // set tolerances
            N_VConst(options.atol, avtol);

            for (&yy_si, &yp_si) in zip(&yy_s, &yp_s) {
                N_VConst(0.0, yy_si);
                N_VConst(0.0, yp_si);
            }
            
            // initialise solver
            Self::check(IDAInit(ida_mem, Some(Self::sresidual), 0.0, yy, yp))?;

            // set tolerances
            Self::check(IDASVtolerances(ida_mem, options.rtol, avtol))?;

            // set events
            //IDARootInit(ida_mem, number_of_events, events_casadi);


            // set matrix
            let jacobian = if options.jacobian == "sparse" {
                return Err(anyhow!("sparse jacobian not implemented"))
            }
            else if options.jacobian == "dense" || options.jacobian == "none" {
                SUNDenseMatrix(i64::from(number_of_states), i64::from(number_of_states))
            }
            else if options.jacobian == "matrix-free" {
                null_mut()
            } else {
                return Err(anyhow!("unknown jacobian {}", options.jacobian))
            };

            let precon_type = i32::try_from(if options.preconditioner == "none" {
                PREC_NONE
            } else {
                PREC_LEFT
            })?;

            // set linear solver
            let linear_solver = if options.linear_solver == "SUNLinSol_Dense" {
                SUNLinSol_Dense(yy, jacobian)
            }
            else if options.linear_solver == "SUNLinSol_KLU" {
                return Err(anyhow!("KLU linear solver not implemented"))
            }
            else if options.linear_solver == "SUNLinSol_SPBCGS" {
                SUNLinSol_SPBCGS(yy, precon_type, i32::try_from(options.linsol_max_iterations)?)
            }
            else if options.linear_solver == "SUNLinSol_SPFGMR" {
                SUNLinSol_SPFGMR(yy, precon_type, i32::try_from(options.linsol_max_iterations)?)
            }
            else if options.linear_solver == "SUNLinSol_SPGMR" {
                SUNLinSol_SPGMR(yy, precon_type, i32::try_from(options.linsol_max_iterations)?)
            }
            else if options.linear_solver == "SUNLinSol_SPTFQMR" {
                SUNLinSol_SPTFQMR(yy, precon_type, i32::try_from(options.linsol_max_iterations)?)
            } else {
                return Err(anyhow!("unknown linear solver {}", options.linear_solver))
            };

            Self::check(IDASetLinearSolver(ida_mem, linear_solver, jacobian))?;

            if options.preconditioner != "none" {
                return Err(anyhow!("preconditioner not implemented"))
            }

            if options.jacobian == "matrix-free" {
                //IDASetJacTimes(ida_mem, null, jtimes);
                todo!()
            }
            else if options.jacobian != "none" {
                //IDASetJacFn(ida_mem, jacobian_casadi);
                todo!()
            }

            if number_of_parameters > 0 {
                //IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
                //            sensitivities, yyS, ypS);
                //IDASensEEtolerances(ida_mem);
            }

            SUNLinSolInitialize(linear_solver);

            let id_val = N_VGetArrayPointer(id);
            for (state_blk, &is_algebraic) in model.state().elmts().iter().zip(model.is_algebraic().iter()) {
                let nnz_start = Layout::ravel_index(state_blk.start(), state_blk.layout().shape());
                let is_alebraic_int = if is_algebraic { 0.0 } else { 1.0 };
                for i in nnz_start..nnz_start + state_blk.nnz() {
                    *id_val.add(i) = is_alebraic_int;
                }
            }

            Self::check(IDASetId(ida_mem, id))?;


            let mut data = Box::new(
                SundialsData {
                    number_of_states: usize::try_from(number_of_states).unwrap(),
                    number_of_parameters: usize::try_from(number_of_parameters).unwrap(),
                    yy,
                    yp,
                    avtol,
                    data,
                    data_layout,
                    yy_s,
                    yp_s,
                    id,
                    jacobian,
                    linear_solver,
                    residual,
                    set_u0,
                    calc_out,
                    options,
                    input_names,
                }
            );
            Self::check(IDASetUserData(ida_mem, &mut *data as *mut _ as *mut c_void))?;
            let sundials = Sundials {
                ida_mem,
                data,
            };
            Ok(sundials)
        }
    }

    pub fn solve(&mut self, times: &Array1<f64>, inputs: &Array1<f64>) -> Result<Array2<f64>> {
        self.set_inputs(inputs).unwrap();
        let number_of_timesteps = times.len();
        let data_out = self.data.get_tensor_data("out").unwrap();
        let mut out_return = Array2::zeros((number_of_timesteps, data_out.len()).f());

        unsafe {
            let data_ptr_mut = self.data.get_data_ptr_mut();

            let yval = N_VGetArrayPointer(self.data.yy);
            let ypval = N_VGetArrayPointer(self.data.yp);
            let mut ys_val: Vec<*mut f64> = Vec::new();
            for is in 0..self.data.number_of_parameters {
                ys_val.push(N_VGetArrayPointer(self.data.yy_s[is]));
                N_VConst(0.0, self.data.yy_s[is]);
                N_VConst(0.0, self.data.yp_s[is]);
            }

            self.data.set_u0.call(data_ptr_mut, self.data.data_layout.indices().as_ptr(), yval, ypval);

            let t0 = times[0];

            Self::check(IDAReInit(self.ida_mem, t0, self.data.yy, self.data.yp))?;

            Self::check(IDACalcIC(self.ida_mem, IDA_YA_YDP_INIT, times[1]))?;

            Self::check(IDAGetConsistentIC(self.ida_mem, self.data.yy, self.data.yp))?;
            
            self.data.calc_out.call(
                t0, 
                N_VGetArrayPointer(self.data.yy), 
                N_VGetArrayPointer(self.data.yp), 
                data_ptr_mut,
                self.data.data_layout.indices().as_ptr(),
            );
            for j in 0..data_out.len() {
                out_return[[0, j]] = data_out[j];
            }
            
            //for j in 0..self.data.number_of_parameters {
            //    for k in 0..self.data.number_of_states {
            //        y_s_return[[0, j, k]] = *ys_val[j].add(k);
            //    }
            //}

            let t_final = times.last().unwrap().clone();
            for t_i in 1..number_of_timesteps {
                let t_next = times[t_i];
                Self::check(IDASetStopTime(self.ida_mem, t_next))?;
                let mut tret: realtype = 0.0;
                let retval: c_int;
                retval = IDASolve(self.ida_mem, t_final, & mut tret as *mut realtype, self.data.yy, self.data.yp, IDA_NORMAL);
                Self::check(retval)?;

                //if self.data.number_of_parameters > 0 {
                //    IDAGetSens(self.ida_mem, & mut tret as *mut realtype, self.data.yy_s.as_mut_ptr());
                //}

                self.data.calc_out.call(
                    tret, 
                    N_VGetArrayPointer(self.data.yy), 
                    N_VGetArrayPointer(self.data.yp), 
                    data_ptr_mut,
                    self.data.data_layout.indices().as_ptr(),
                );
                for j in 0..data_out.len() {
                    out_return[[t_i, j]] = data_out[j];
                }
                //for j in 0..self.data.number_of_parameters {
                //    for k in 0..self.data.number_of_states {
                //        y_s_return[[t_i, j, k]] = *ys_val[j].add(k);
                //    }
                //}
                if retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN {
                    break;
                }
            }

            if self.data.options.print_stats {
                let mut nsteps = 0_i64;
                let mut nrevals = 0_i64;
                let mut nlinsetups = 0_i64;
                let mut netfails = 0_i64;
                let mut klast = 0_i32;
                let mut kcur = 0_i32;

                let mut hinused = 0.0;
                let mut hlast = 0.0;
                let mut hcur = 0.0;
                let mut tcur = 0.0;

                Self::check(IDAGetIntegratorStats(self.ida_mem, 
                                    &mut nsteps as *mut i64, 
                                    &mut nrevals as *mut i64, 
                                    &mut nlinsetups as *mut i64, 
                                    &mut netfails as *mut i64,
                                    &mut klast as *mut i32, 
                                    &mut kcur as *mut i32, 
                                    &mut hinused as *mut f64, 
                                    &mut hlast as *mut f64,
                                    &mut hcur as *mut f64, 
                                    &mut tcur as *mut f64))?;

                    
                let mut nniters = 0_i64;
                let mut nncfails = 0_i64;
                Self::check(IDAGetNonlinSolvStats(self.ida_mem, &mut nniters as *mut i64, &mut nncfails as *mut i64))?;

                //let ngevalsBBDP = 0;
                //if false {
                //    IDABBDPrecGetNumGfnEvals(ida_mem, &ngevalsBBDP);
                //}

                println!("Solver Stats:");
                println!("\tNumber of steps = {}", nsteps);
                println!("\tNumber of calls to residual function = {}", nrevals);
                //println!("\tNumber of calls to residual function in preconditioner = {}",
                //        ngevalsBBDP);
                println!("\tNumber of linear solver setup calls = {}", nlinsetups);
                println!("\tNumber of error test failures = {}", netfails);
                println!("\tMethod order used on last step = {}", klast);
                println!("\tMethod order used on next step = {}", kcur);
                println!("\tInitial step size = {}", hinused);
                println!("\tStep size on last step = {}", hlast);
                println!("\tStep size on next step = {}", hcur);
                println!("\tCurrent internal time reached = {}", tcur);
                println!("\tNumber of nonlinear iterations performed = {}", nniters);
                println!("\tNumber of nonlinear convergence failures = {}", nncfails);
            }

        }

        return Ok(out_return);
    }

    pub fn destroy(&mut self) {
        unsafe {
             /* Free memory */
            //if self.data.number_of_parameters > 0 {
            //    IDASensFree(self.ida_mem);
            //}
            SUNLinSolFree(self.data.linear_solver);
            SUNMatDestroy(self.data.jacobian);
            N_VDestroy(self.data.avtol);
            N_VDestroy(self.data.yy);
            N_VDestroy(self.data.yp);
            N_VDestroy(self.data.id);
            for (&yy_si, &yp_si) in zip(&self.data.yy_s, &self.data.yp_s) {
                N_VDestroy(yy_si);
                N_VDestroy(yp_si);
            }
            IDAFree(&mut self.ida_mem as *mut *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
use approx::assert_relative_eq;
use ndarray::{Array, array, s};

use crate::{ms_parser::parse_string, discretise::{DiscreteModel, Translation}, builder::ModelInfo, codegen::{Sundials, Options}, ds_parser};

    macro_rules! translation_test {
        ($($name:ident: $text:literal expect $blk_name:literal = $expected_value:expr,)*) => {
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
                let model = ds_parser::parse_string(full_text.as_str()).unwrap();
                let discrete_model = match DiscreteModel::build("$name", &model) {
                    Ok(model) => {
                        model
                    }
                    Err(e) => {
                        panic!("{}", e.as_error_message(full_text.as_str()));
                    }
                };
                let tensor = discrete_model.time_indep_defns().iter().find(|t| t.elmts().iter().find(|blk| blk.name() == Some($blk_name)).is_some()).unwrap();
                let blk = tensor.elmts().iter().find(|blk| blk.name() == Some($blk_name)).unwrap();
                let translation = Translation::new(blk.expr_layout(), blk.layout(), &blk.start(), tensor.layout_ptr());
                assert_eq!(translation.to_string(), $expected_value);
            }
        )*
        }
    }

    translation_test!{
        elementwise_scalar: "r { y = 2}" expect "y" = "Translation(ElementWise, Contiguous(0, 1))",
        elementwise_vector: "r_i { 1, y = 2}" expect "y" = "Translation(Broadcast(1, 1), Contiguous(1, 2))",
        elementwise_vector2: "a_i { 1, 2 } r_i { 1, y = a_i}" expect "y" = "Translation(ElementWise, Contiguous(1, 3))",
        broadcast_by_1: "r_i { (0:4): y = 2}" expect "y" = "Translation(Broadcast(1, 4), Contiguous(0, 4))",
        broadcast_by_2: "r_ij { (0:4, 0:3): y = 2}" expect "y" = "Translation(Broadcast(2, 12), Contiguous(0, 12))",
        sparse_rearrange_23: "r_ij { (0, 0): 1, (1, 1): y = 2, (0, 1): 3 }" expect "y" = "Translation(Broadcast(2, 1), Contiguous(2, 3))",
        sparse_rearrange_12: "r_ij { (0, 0): 1, (1, 1): 2, (0, 1): y = 3 }" expect "y" = "Translation(Broadcast(2, 1), Contiguous(1, 2))",
        contiguous_in_middle: "r_i { 1, (1:5): y = 2, 2, 3}" expect "y" = "Translation(Broadcast(1, 4), Contiguous(1, 5))",
        dense_to_contiguous_sparse: "A_ij { (0, 0): 1, (1, 1): y = 2, (0, 1): 3 }" expect "y" = "Translation(Broadcast(2, 1), Contiguous(2, 3))",
        dense_to_sparse_sparse: "A_ij { (0, 0): 1, (1:4, 1): y = 2, (2, 2): 1, (4, 4): 3 }" expect "y" = "Translation(Broadcast(2, 3), Sparse[1, 2, 4])",
        dense_to_sparse_sparse2: "A_ij { (0, 0): 1, (1:4, 1): y = 2, (1, 2): 1, (4, 4): 3 }" expect "y" = "Translation(Broadcast(2, 3), Sparse[1, 3, 4])",
        sparse_contraction: "A_ij { (0, 0): 1, (1, 1): 2, (0, 1): 3 } b_i { 1, 2 } x_i { y = A_ij * b_j }" expect "y" = "Translation(SparseContraction(1, [0, 2], [2, 3]), Contiguous(0, 2))",
        dense_contraction: "A_ij { (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 2 } b_i { 1, 2 } x_i { y = A_ij * b_j }" expect "y" = "Translation(DenseContraction(1, 2), Contiguous(0, 2))",
        diagonal_contraction: "A_ij { (0..2, 0..2): 1 } b_i { 1, 2 } x_i { y = A_ij * b_j }" expect "y" = "Translation(DiagonalContraction(1), Contiguous(0, 2))",
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
                let model = ds_parser::parse_string(full_text.as_str()).unwrap();
                let options = Options::new();
                let context = inkwell::context::Context::create();
                let discrete_model = match DiscreteModel::build("$name", &model) {
                    Ok(model) => {
                        model
                    }
                    Err(e) => {
                        panic!("{}", e.as_error_message(full_text.as_str()));
                    }
                };
                let mut sundials = Sundials::from_discrete_model(&discrete_model, &context, options).unwrap();
                let inputs = array![];
                let u0 = array![1.];
                let up0 = array![1.];
                let check_u0 = sundials.calc_u0(&inputs);
                assert_relative_eq!(u0, check_u0);
                let tensor = sundials.calc_tensor($tensor_name, 0., &inputs, &u0, &up0);
                assert_relative_eq!(tensor, $expected_value);
            }
        )*
        }
    }

    tensor_test!{
        scalar: "r {2}" expect "r" array![2.0,],
        constant: "r_i {2, 3}" expect "r" array![2., 3.],
        expression: "r_i {2 + 3, 3 * 2}" expect "r" array![5., 6.],
        derived: "r_i {2, 3} k_i { 2 * r_i }" expect "k" array![4., 6.],
        concatenate: "r_i {2, 3} k_i { r_i, 2 * r_i }" expect "k" array![2., 3., 4., 6.],
        ones_matrix_dense: "I_ij { (0:2, 0:2): 1 }" expect "I" array![1., 1., 1., 1.],
        dense_matrix: "A_ij { (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 }" expect "A" array![1., 2., 3., 4.],
        identity_matrix_diagonal: "I_ij { (0..2, 0..2): 1 }" expect "I" array![1., 1.],
        concatenate_diagonal: "A_ij { (0..2, 0..2): 1 } B_ij { (0:2, 0:2): A_ij, (2:4, 2:4): A_ij }" expect "B" array![1., 1., 1., 1.],
        identity_matrix_sparse: "I_ij { (0, 0): 1, (1, 1): 2 }" expect "I" array![1., 2.],
        concatenate_sparse: "A_ij { (0, 0): 1, (1, 1): 2 } B_ij { (0:2, 0:2): A_ij, (2:4, 2:4): A_ij }" expect "B" array![1., 2., 1., 2.],
        sparse_rearrange: "A_ij { (0, 0): 1, (1, 1): 2, (0, 1): 3 }" expect "A" array![1., 3., 2.],
        sparse_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 0): 2, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" array![1., 7.],
        diag_matrix_vect_multiply: "A_ij { (0, 0): 1, (1, 1): 3 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" array![1., 6.],
        dense_matrix_vect_multiply: "A_ij {  (0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4 } x_i { 1, 2 } b_i { A_ij * x_j }" expect "b" array![5., 11.],
    }

     #[test]
    fn rate_equationn() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), z(t)) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
            z = 2 * y
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        println!("{}", discrete);
        let options = Options::new();
        let context = inkwell::context::Context::create();
        let mut sundials = Sundials::from_discrete_model(&discrete, &context, options).unwrap();

        let times = Array::linspace(0., 1., 5);


        let y0 = 1.0;
        let inputs = array![1., 1.];
        let u0 = sundials.calc_u0(&inputs);
        let check_u0 = array![y0, 0.];
        assert_relative_eq!(u0, check_u0);

        for r in Array::linspace(0.1, 2.0, 5) {
            for k in Array::linspace(0.1, 2.0, 5) {
                let inputs = array![r, k];
                for y in Array::linspace(0.1, 2.0, 5) {
                    for z in Array::linspace(0.1, 2.0, 5) {
                        let u0 = array![y, z];
                        for dy in Array::linspace(0.1, 2.0, 5) {
                            for dz in  Array::linspace(0.1, 2.0, 5) {
                                let up0 = array![dy, dz];

                                let f = sundials.calc_tensor("F", 0., &inputs, &u0, &up0);
                                let f_check = array![up0[0], 0.0];
                                assert_relative_eq!(f, f_check);

                                let g = sundials.calc_tensor("G", 0., &inputs, &u0, &up0);
                                let g_check = array![
                                    (r * u0[0]) * (1.0 - (u0[0] / k)),
                                    (2.0 * u0[0]) - u0[1],
                                ];
                                assert_relative_eq!(g, g_check);

                                // test residual
                                let res = sundials.calc_residual(0., &inputs, &u0, &up0);
                                let res_check = array![
                                    up0[0] - (r * u0[0]) * (1.0 - (u0[0] / k)),
                                    0.0 - (2.0 * u0[0]) + u0[1],
                                ];
                                assert_relative_eq!(res, res_check);

                                // test out
                                let out = sundials.calc_out(0., &inputs, &u0, &up0);
                                let out_check = u0.clone();
                                assert_relative_eq!(out, out_check);
                            }
                        }
                    }
                }
            }
        }


        // solve
        let r = 1.0;
        let k = 2.0;
        let inputs = array![r, k];
        let out = sundials.solve(&times, &inputs).unwrap();

        let y_check = k / ((k - y0) * (-r * times).mapv(f64::exp) / y0 + 1.);
        assert_relative_eq!(y_check, out.slice(s![.., 0]), epsilon=1e-5);
        assert_relative_eq!(y_check * 2., out.slice(s![.., 1]), epsilon=1e-5);

        sundials.destroy();
    }
}
 