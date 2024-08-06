use anyhow::{anyhow, Result};
use codegen::ir::{FuncRef, StackSlot};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};
use std::collections::HashMap;
use std::slice;

use crate::ast::{Ast, AstKind};
use crate::discretise::{Tensor, TensorBlock};
use crate::execution::{DataLayout, Translation, TranslationFrom};



/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct CraneliftCodeGen<'a> {
    int_type: types::Type,
    real_type: types::Type,
    real_ptr_type: types::Type,
    int_ptr_type: types::Type,
    builder: FunctionBuilder<'a>,
    variable_index: usize,
    module: &'a mut JITModule,
    tensor_ptr: Option<Value>,
    tensors: HashMap<String, Value>,
    mem_flags: MemFlags,
    functions: HashMap<String, FuncRef>,
    layout: DataLayout,
}

impl<'ctx> CraneliftCodeGen<'ctx> {
    fn jit_compile_expr(
        &mut self,
        name: &str,
        expr: &Ast,
        index: &[Value],
        elmt: &TensorBlock,
        expr_index: Option<Value>,
    ) -> Result<Value> {
       let name = elmt.name().unwrap_or(name);
        match &expr.kind {
            AstKind::Binop(binop) => {
                let lhs =
                    self.jit_compile_expr(name, binop.left.as_ref(), index, elmt, expr_index)?;
                let rhs =
                    self.jit_compile_expr(name, binop.right.as_ref(), index, elmt, expr_index)?;
                match binop.op {
                    '*' => Ok(self.builder.ins().fmul(lhs, rhs)),
                    '/' => Ok(self.builder.ins().fdiv(lhs, rhs)),
                    '-' => Ok(self.builder.ins().fsub(lhs, rhs)),
                    '+' => Ok(self.builder.ins().fadd(lhs, rhs)),
                    unknown => Err(anyhow!("unknown binop op '{}'", unknown)),
                }
            }
            AstKind::Monop(monop) => {
                let child =
                    self.jit_compile_expr(name, monop.child.as_ref(), index, elmt, expr_index)?;
                match monop.op {
                    '-' => Ok(self.builder.ins().fneg(child)),
                    unknown => Err(anyhow!("unknown monop op '{}'", unknown)),
                }
            }
            AstKind::Call(call) => match self.get_function(call.fn_name) {
                Some(function) => {
                    let mut args = Vec::new();
                    for arg in call.args.iter() {
                        let arg_val =
                            self.jit_compile_expr(name, arg.as_ref(), index, elmt, expr_index)?;
                        args.push(arg_val);
                    }
                    let call = self.builder.ins().call(function, &args);
                    let ret_value = self.builder.inst_results(call)[0];
                    Ok(ret_value)
                }
                None => Err(anyhow!("unknown function call '{}'", call.fn_name)),
            },
            AstKind::CallArg(arg) => {
                self.jit_compile_expr(name, &arg.expression, index, elmt, expr_index)
            }
            AstKind::Number(value) => Ok(
                match self.real_type {
                    types::F32 => self.builder.ins().f32const(*value as f32),
                    types::F64 => self.builder.ins().f64const(*value as f64),
                    _ => panic!("unexpected real type"),
                }
            ),
            AstKind::IndexedName(iname) => {
                let ptr = self.variables.get(name).unwrap();
                let layout = self.layout.get_layout(iname.name).unwrap();
                let iname_elmt_index = if layout.is_dense() {
                    // permute indices based on the index chars of this tensor
                    let mut no_transform = true;
                    let mut iname_index = Vec::new();
                    for (i, c) in iname.indices.iter().enumerate() {
                        // find the position index of this index char in the tensor's index chars,
                        // if it's not found then it must be a contraction index so is at the end
                        let pi = elmt
                            .indices()
                            .iter()
                            .position(|x| x == c)
                            .unwrap_or(elmt.indices().len());
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
                            let stride_intval = self.builder.ins().iconst(self.int_type, i64::try_from(stride).unwrap());
                            let stride_mul_i = self.builder.ins().imul(stride_intval, iname_i);
                            let iname_elmt_index =
                                self.builder.ins().iadd(iname_elmt_index, stride_mul_i);
                        }
                        Some(iname_elmt_index)
                    } else {
                        let zero = self.builder.ins().iconst(self.int_type, 0);
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
                    Some(offset) => self.builder.ins().iadd(*ptr, offset),
                    None => *ptr,
                };
                Ok(self.builder.ins().load(self.real_type, self.mem_flags, value_ptr, 0))
            }
            AstKind::Name(name) => {
                // must be a scalar, just load the value
                let ptr = self.variables.get(*name).unwrap();
                Ok(self.builder.ins().load(self.real_type, self.mem_flags, *ptr, 0))
            }
            AstKind::NamedGradient(name) => {
                let name_str = name.to_string();
                let ptr = self.variables.get(&name_str).unwrap();
                Ok(self.builder.ins().load(self.real_type, self.mem_flags, *ptr, 0))
            }
            AstKind::Index(_) => todo!(),
            AstKind::Slice(_) => todo!(),
            AstKind::Integer(_) => todo!(),
            _ => panic!("unexprected astkind"),
        }
    }

    fn get_function(&mut self, name: &str) -> Option<FuncRef> {
        match self.functions.get(name) {
            Some(&func) => Some(func),
            None => {
                let function = match name {
                    // support some standard library functions
                    "sin" | "cos" | "tan" | "exp" | "log" | "log10" | "sqrt" | "abs"
                    | "copysign" | "pow" | "min" | "max" => {
                        let args = match name {
                            "pow" => vec![self.real_type, self.real_type],
                            _ => vec![self.real_type],
                        };
                        let ret_type = self.real_type;

                        let mut sig = self.module.make_signature();
                        for arg in &args {
                            sig.params.push(AbiParam::new(*arg));
                        }
                        sig.returns.push(AbiParam::new(ret_type));

                        let callee = self
                            .module
                            .declare_function(&name, Linkage::Import, &sig)
                            .expect("problem declaring function");
                        Some(self.module.declare_func_in_func(callee, self.builder.func))
                    },
                    _ => None,
                }?;
                self.functions.insert(name.to_owned(), function);
                Some(function)
            }
        }
    }
    
    fn jit_compile_tensor(
        &mut self,
        a: &Tensor,
    ) -> Result<Value> {
        // set up the tensor storage pointer and index into this data
        let res_ptr = *self.tensors.get(a.name()).expect(format!("tensor {} not defined", a.name()).as_str());
        self.tensor_ptr = Some(res_ptr);

        // reset variable index to 0
        self.variable_index = 0;

        // treat scalar as a special case
        if a.rank() == 0 {

            let elmt = a.elmts().first().unwrap();
            let float_value = self.jit_compile_expr(a.name(), elmt.expr(), &[], elmt, None)?;
            self.builder.ins().store(self.mem_flags, float_value, res_ptr, 0);
        }

        for (i, blk) in a.elmts().iter().enumerate() {
            let default = format!("{}-{}", a.name(), i);
            let name = blk.name().unwrap_or(default.as_str());
            self.jit_compile_block(name, a, blk)?;
        }
        Ok(res_ptr)
    }

     fn jit_compile_block(&mut self, name: &str, tensor: &Tensor, elmt: &TensorBlock) -> Result<()> {
        let translation = Translation::new(
            elmt.expr_layout(),
            elmt.layout(),
            elmt.start(),
            tensor.layout_ptr(),
        );

        if elmt.expr_layout().is_dense() {
            self.jit_compile_dense_block(name, elmt, &translation)
        } else if elmt.expr_layout().is_diagonal() {
            self.jit_compile_diagonal_block(name, elmt, &translation)
        } else if elmt.expr_layout().is_sparse() {
            match translation.source {
                TranslationFrom::SparseContraction { .. } => {
                    self.jit_compile_sparse_contraction_block(name, elmt, &translation)
                }
                _ => self.jit_compile_sparse_block(name, elmt, &translation),
            }
        } else {
            return Err(anyhow!(
                "unsupported block layout: {:?}",
                elmt.expr_layout()
            ));
        }
    }

    fn decl_stack_slot(&mut self, ty: Type, val: Option<Value>) -> StackSlot {
        let data = StackSlotData::new(StackSlotKind::ExplicitSlot, ty.bytes(), 0);
        let ss = self.builder.create_sized_stack_slot(data);
        if let Some(val) = val {
            self.builder.ins().stack_store(val, ss, 0);
        }
        ss
    }

      // for dense blocks we can loop through the nested loops to calculate the index, then we compile the expression passing in this index
    fn jit_compile_dense_block(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        translation: &Translation,
    ) -> Result<()> {
        let int_type = self.int_type;

        let mut preblock = self.builder.create_block();
        let expr_rank = elmt.expr_layout().rank();
        let expr_shape = elmt
            .expr_layout()
            .shape()
            .mapv(|n| i64::try_from(n).unwrap());
        let one = self.builder.ins().iconst(int_type, 1);
        let zero = self.builder.ins().iconst(int_type, 0);

        let expr_index_var = self.decl_stack_slot(self.int_type, Some(zero));
        let elmt_index_var = self.decl_stack_slot(self.int_type, Some(zero));

        // setup indices, loop through the nested loops
        let mut indices = Vec::new();
        let mut blocks = Vec::new();

        // allocate the contract sum if needed
        let (contract_sum, contract_by) = if let TranslationFrom::DenseContraction {
            contract_by,
            contract_len: _,
        } = translation.source
        {
            (
                Some(self.decl_stack_slot(self.real_type, None)),
                contract_by,
            )
        } else {
            (None, 0)
        };

        for i in 0..expr_rank {
            let block = self.builder.create_block();
            let curr_index = self.builder.append_block_param(block, self.int_type);
            self.builder.ins().jump(block, &[zero]);
            self.builder.switch_to_block(block);

            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                self.builder.ins().stack_store(zero, contract_sum.unwrap(), 0);
            }

            indices.push(curr_index);
            blocks.push(block);
            preblock = block;
        }

        let elmt_index = self.builder.ins().stack_load(self.int_type, elmt_index_var, 0);

        // load and increment the expression index
        let expr_index = self.builder.ins().stack_load(self.int_type, expr_index_var, 0);
        let next_expr_index = self.builder.ins().iadd(expr_index, one);
        self.builder.ins().stack_store(next_expr_index, expr_index_var, 0);

        let float_value = self.jit_compile_expr(
            name,
            elmt.expr(),
            indices.as_slice(),
            elmt,
            Some(expr_index),
        )?;

        if contract_sum.is_some() {
            let contract_sum_value = self
                .builder
                .ins()
                .stack_load(self.real_type, contract_sum.unwrap(), 0);
            let new_contract_sum_value = self.builder.ins().fadd(contract_sum_value, float_value);
            self.builder
                .ins()
                .stack_store(new_contract_sum_value, contract_sum.unwrap(), 0);
        } else {
            preblock = self.jit_compile_broadcast_and_store(
                name,
                elmt,
                float_value,
                expr_index,
                translation,
                preblock,
            )?;
            let next_elmt_index = self.builder.ins().iadd(elmt_index, one);
            self.builder.ins().stack_store(next_elmt_index, elmt_index_var, 0);
        }

        // unwind the nested loops
        for i in (0..expr_rank).rev() {

            // update and store contract sum
            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                let next_elmt_index = self.builder.ins().iadd(elmt_index, one);
                self.builder.ins().stack_store(next_elmt_index, elmt_index_var, 0);

                let contract_sum_value = self
                    .builder
                    .ins()
                    .stack_load(self.real_type, contract_sum.unwrap(), 0);
                
                self.jit_compile_store(name, elmt, elmt_index, contract_sum_value, translation)?;
            }


            // increment index
            let next_index = self.builder.ins().iadd(indices[i], one);
            let block = self.builder.create_block();
            self.builder.ins().brif(
                self.builder.ins().icmp_imm(IntCC::UnsignedLessThan, next_index, expr_shape[i]),
                blocks[i],
                &[next_index],
                block,
                &[],
            );
            self.builder.seal_block(blocks[i]);
            self.builder.seal_block(block);
            self.builder.switch_to_block(block);
            preblock = block;
        }
        Ok(())
    }

      fn jit_compile_sparse_contraction_block(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        translation: &Translation,
    ) -> Result<()> {
        match translation.source {
            TranslationFrom::SparseContraction { .. } => {}
            _ => {
                panic!("expected sparse contraction")
            }
        }
        let int_type = self.int_type;

        let preblock = self.builder.get_insert_block().unwrap();
        let layout_index = self.layout.get_layout_index(elmt.expr_layout()).unwrap();
        let translation_index = self
            .layout
            .get_translation_index(elmt.expr_layout(), elmt.layout())
            .unwrap();
        let translation_index = translation_index + translation.get_from_index_in_data_layout();

        let contract_sum_ptr = self.builder.build_alloca(self.real_type, "contract_sum")?;

        // loop through each contraction
        let block = self.context.append_basic_block(self.fn_value(), name);
        self.builder.build_unconditional_branch(block)?;
        self.builder.position_at_end(block);

        let contract_index = self.builder.build_phi(int_type, "i")?;
        let final_contract_index =
            int_type.const_int(elmt.layout().nnz().try_into().unwrap(), false);
        contract_index.add_incoming(&[(&int_type.const_int(0, false), preblock)]);

        let start_index = self.builder.build_int_add(
            int_type.const_int(translation_index.try_into().unwrap(), false),
            self.builder.build_int_mul(
                int_type.const_int(2, false),
                contract_index.as_basic_value().into_int_value(),
                name,
            )?,
            name,
        )?;
        let end_index =
            self.builder
                .build_int_add(start_index, int_type.const_int(1, false), name)?;
        let start_ptr = self.build_gep(
            self.int_type,
            *self.get_param("indices"),
            &[start_index],
            "start_index_ptr",
        )?;
        let start_contract = self
            .build_load(self.int_type, start_ptr, "start")?
            .into_int_value();
        let end_ptr = self.build_gep(
            self.int_type,
            *self.get_param("indices"),
            &[end_index],
            "end_index_ptr",
        )?;
        let end_contract = self
            .build_load(self.int_type, end_ptr, "end")?
            .into_int_value();

        // initialise the contract sum
        self.builder
            .build_store(contract_sum_ptr, self.real_type.const_float(0.0))?;

        // loop through each element in the contraction
        let contract_block = self
            .context
            .append_basic_block(self.fn_value(), format!("{}_contract", name).as_str());
        self.builder.build_unconditional_branch(contract_block)?;
        self.builder.position_at_end(contract_block);

        let expr_index_phi = self.builder.build_phi(int_type, "j")?;
        expr_index_phi.add_incoming(&[(&start_contract, block)]);

        // loop body - load index from layout
        let expr_index = expr_index_phi.as_basic_value().into_int_value();
        let elmt_index_mult_rank = self.builder.build_int_mul(
            expr_index,
            int_type.const_int(elmt.expr_layout().rank().try_into().unwrap(), false),
            name,
        )?;
        let indices_int = (0..elmt.expr_layout().rank())
            .map(|i| {
                let layout_index_plus_offset =
                    int_type.const_int((layout_index + i).try_into().unwrap(), false);
                let curr_index = self.builder.build_int_add(
                    elmt_index_mult_rank,
                    layout_index_plus_offset,
                    name,
                )?;
                let ptr = Self::get_ptr_to_index(
                    &self.builder,
                    self.int_type,
                    self.get_param("indices"),
                    curr_index,
                    name,
                );
                let index = self.build_load(self.int_type, ptr, name)?.into_int_value();
                Ok(index)
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        // loop body - eval expression and increment sum
        let float_value = self.jit_compile_expr(
            name,
            elmt.expr(),
            indices_int.as_slice(),
            elmt,
            Some(expr_index),
        )?;
        let contract_sum_value = self
            .build_load(self.real_type, contract_sum_ptr, "contract_sum")?
            .into_float_value();
        let new_contract_sum_value =
            self.builder
                .build_float_add(contract_sum_value, float_value, "new_contract_sum")?;
        self.builder
            .build_store(contract_sum_ptr, new_contract_sum_value)?;

        // increment contract loop index
        let next_elmt_index =
            self.builder
                .build_int_add(expr_index, int_type.const_int(1, false), name)?;
        expr_index_phi.add_incoming(&[(&next_elmt_index, contract_block)]);

        // contract loop condition
        let loop_while = self.builder.build_int_compare(
            IntPredicate::ULT,
            next_elmt_index,
            end_contract,
            name,
        )?;
        let post_contract_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder
            .build_conditional_branch(loop_while, contract_block, post_contract_block)?;
        self.builder.position_at_end(post_contract_block);

        // store the result
        self.jit_compile_store(
            name,
            elmt,
            contract_index.as_basic_value().into_int_value(),
            new_contract_sum_value,
            translation,
        )?;

        // increment outer loop index
        let next_contract_index = self.builder.build_int_add(
            contract_index.as_basic_value().into_int_value(),
            int_type.const_int(1, false),
            name,
        )?;
        contract_index.add_incoming(&[(&next_contract_index, post_contract_block)]);

        // outer loop condition
        let loop_while = self.builder.build_int_compare(
            IntPredicate::ULT,
            next_contract_index,
            final_contract_index,
            name,
        )?;
        let post_block = self.context.append_basic_block(self.fn_value(), name);
        self.builder
            .build_conditional_branch(loop_while, block, post_block)?;
        self.builder.position_at_end(post_block);

        Ok(())
    }




    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: Expr) -> Value {
        match expr {
            Expr::Literal(literal) => {
                let imm: i32 = literal.parse().unwrap();
                self.builder.ins().iconst(self.int, i64::from(imm))
            }

            Expr::Add(lhs, rhs) => {
                let lhs = self.translate_expr(*lhs);
                let rhs = self.translate_expr(*rhs);
                self.builder.ins().iadd(lhs, rhs)
            }

            Expr::Sub(lhs, rhs) => {
                let lhs = self.translate_expr(*lhs);
                let rhs = self.translate_expr(*rhs);
                self.builder.ins().isub(lhs, rhs)
            }

            Expr::Mul(lhs, rhs) => {
                let lhs = self.translate_expr(*lhs);
                let rhs = self.translate_expr(*rhs);
                self.builder.ins().imul(lhs, rhs)
            }

            Expr::Div(lhs, rhs) => {
                let lhs = self.translate_expr(*lhs);
                let rhs = self.translate_expr(*rhs);
                self.builder.ins().udiv(lhs, rhs)
            }

            Expr::Eq(lhs, rhs) => self.translate_icmp(IntCC::Equal, *lhs, *rhs),
            Expr::Ne(lhs, rhs) => self.translate_icmp(IntCC::NotEqual, *lhs, *rhs),
            Expr::Lt(lhs, rhs) => self.translate_icmp(IntCC::SignedLessThan, *lhs, *rhs),
            Expr::Le(lhs, rhs) => self.translate_icmp(IntCC::SignedLessThanOrEqual, *lhs, *rhs),
            Expr::Gt(lhs, rhs) => self.translate_icmp(IntCC::SignedGreaterThan, *lhs, *rhs),
            Expr::Ge(lhs, rhs) => self.translate_icmp(IntCC::SignedGreaterThanOrEqual, *lhs, *rhs),
            Expr::Call(name, args) => self.translate_call(name, args),
            Expr::GlobalDataAddr(name) => self.translate_global_data_addr(name),
            Expr::Identifier(name) => {
                // `use_var` is used to read the value of a variable.
                let variable = self.variables.get(&name).expect("variable not defined");
                self.builder.use_var(*variable)
            }
            Expr::Assign(name, expr) => self.translate_assign(name, *expr),
            Expr::IfElse(condition, then_body, else_body) => {
                self.translate_if_else(*condition, then_body, else_body)
            }
            Expr::WhileLoop(condition, loop_body) => {
                self.translate_while_loop(*condition, loop_body)
            }
        }
    }

    fn translate_assign(&mut self, name: String, expr: Expr) -> Value {
        // `def_var` is used to write the value of a variable. Note that
        // variables can have multiple definitions. Cranelift will
        // convert them into SSA form for itself automatically.
        let new_value = self.translate_expr(expr);
        let variable = self.variables.get(&name).unwrap();
        self.builder.def_var(*variable, new_value);
        new_value
    }

    fn translate_icmp(&mut self, cmp: IntCC, lhs: Expr, rhs: Expr) -> Value {
        let lhs = self.translate_expr(lhs);
        let rhs = self.translate_expr(rhs);
        self.builder.ins().icmp(cmp, lhs, rhs)
    }

    fn translate_if_else(
        &mut self,
        condition: Expr,
        then_body: Vec<Expr>,
        else_body: Vec<Expr>,
    ) -> Value {
        let condition_value = self.translate_expr(condition);

        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If-else constructs in the toy language have a return value.
        // In traditional SSA form, this would produce a PHI between
        // the then and else bodies. Cranelift uses block parameters,
        // so set up a parameter in the merge block, and we'll pass
        // the return values to it from the branches.
        self.builder.append_block_param(merge_block, self.int);

        // Test the if condition and conditionally branch.
        self.builder
            .ins()
            .brif(condition_value, then_block, &[], else_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        let mut then_return = self.builder.ins().iconst(self.int, 0);
        for expr in then_body {
            then_return = self.translate_expr(expr);
        }

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &[then_return]);

        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);
        let mut else_return = self.builder.ins().iconst(self.int, 0);
        for expr in else_body {
            else_return = self.translate_expr(expr);
        }

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &[else_return]);

        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);

        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);

        // Read the value of the if-else by reading the merge block
        // parameter.
        let phi = self.builder.block_params(merge_block)[0];

        phi
    }

    fn translate_while_loop(&mut self, condition: Expr, loop_body: Vec<Expr>) -> Value {
        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);
        self.builder.switch_to_block(header_block);

        let condition_value = self.translate_expr(condition);
        self.builder
            .ins()
            .brif(condition_value, body_block, &[], exit_block, &[]);

        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        for expr in loop_body {
            self.translate_expr(expr);
        }
        self.builder.ins().jump(header_block, &[]);

        self.builder.switch_to_block(exit_block);

        // We've reached the bottom of the loop, so there will be no
        // more backedges to the header to exits to the bottom.
        self.builder.seal_block(header_block);
        self.builder.seal_block(exit_block);

        // Just return 0 for now.
        self.builder.ins().iconst(self.int, 0)
    }

    fn translate_call(&mut self, name: String, args: Vec<Expr>) -> Value {
        let mut sig = self.module.make_signature();

        // Add a parameter for each argument.
        for _arg in &args {
            sig.params.push(AbiParam::new(self.int));
        }

        // For simplicity for now, just make all calls return a single I64.
        sig.returns.push(AbiParam::new(self.int));

        // TODO: Streamline the API here?
        let callee = self
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .expect("problem declaring function");
        let local_callee = self.module.declare_func_in_func(callee, self.builder.func);

        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.translate_expr(arg))
        }
        let call = self.builder.ins().call(local_callee, &arg_values);
        self.builder.inst_results(call)[0]
    }

    fn translate_global_data_addr(&mut self, name: String) -> Value {
        let sym = self
            .module
            .declare_data(&name, Linkage::Export, true, false)
            .expect("problem declaring data object");
        let local_id = self.module.declare_data_in_func(sym, self.builder.func);

        let pointer = self.module.target_config().pointer_type();
        self.builder.ins().symbol_value(pointer, local_id)
    }
}

fn declare_variables(
    int: types::Type,
    builder: &mut FunctionBuilder,
    params: &[String],
    the_return: &str,
    stmts: &[Expr],
    entry_block: Block,
) -> HashMap<String, Variable> {
    let mut variables = HashMap::new();
    let mut index = 0;

    for (i, name) in params.iter().enumerate() {
        // TODO: cranelift_frontend should really have an API to make it easy to set
        // up param variables.
        let val = builder.block_params(entry_block)[i];
        let var = declare_variable(int, builder, &mut variables, &mut index, name);
        builder.def_var(var, val);
    }
    let zero = builder.ins().iconst(int, 0);
    let return_variable = declare_variable(int, builder, &mut variables, &mut index, the_return);
    builder.def_var(return_variable, zero);
    for expr in stmts {
        declare_variables_in_stmt(int, builder, &mut variables, &mut index, expr);
    }

    variables
}

/// Recursively descend through the AST, translating all implicit
/// variable declarations.
fn declare_variables_in_stmt(
    int: types::Type,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, Variable>,
    index: &mut usize,
    expr: &Expr,
) {
    match *expr {
        Expr::Assign(ref name, _) => {
            declare_variable(int, builder, variables, index, name);
        }
        Expr::IfElse(ref _condition, ref then_body, ref else_body) => {
            for stmt in then_body {
                declare_variables_in_stmt(int, builder, variables, index, stmt);
            }
            for stmt in else_body {
                declare_variables_in_stmt(int, builder, variables, index, stmt);
            }
        }
        Expr::WhileLoop(ref _condition, ref loop_body) => {
            for stmt in loop_body {
                declare_variables_in_stmt(int, builder, variables, index, stmt);
            }
        }
        _ => (),
    }
}

/// Declare a single variable declaration.
fn declare_variable(
    int: types::Type,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, Variable>,
    index: &mut usize,
    name: &str,
) -> Variable {
    let var = Variable::new(*index);
    if !variables.contains_key(name) {
        variables.insert(name.into(), var);
        builder.declare_var(var, int);
        *index += 1;
    }
    var
}