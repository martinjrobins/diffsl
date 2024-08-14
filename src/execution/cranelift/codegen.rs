use anyhow::{anyhow, Ok, Result};
use codegen::ir::{FuncRef, StackSlot};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use target_lexicon::{PointerWidth, Triple};
use std::collections::HashMap;
use std::iter::zip;

use crate::ast::{Ast, AstKind};
use crate::discretise::{DiscreteModel, Tensor, TensorBlock};
use crate::execution::{DataLayout, Translation, TranslationFrom, TranslationTo};

struct CraneliftModule {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The data description, which is to data objects what `ctx` is to functions.
    data_description: DataDescription,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    module: JITModule,

    layout: DataLayout,

    triple: Triple,

    int_type: types::Type,
    real_type: types::Type,
    real_ptr_type: types::Type,
    int_ptr_type: types::Type,

             
}


impl CraneliftModule {
    fn new(triple: Triple, model: &DiscreteModel) -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let module = JITModule::new(builder);

        let ptr_type = match triple.pointer_width().unwrap() {
            PointerWidth::U16 => types::I16,
            PointerWidth::U32 => types::I32,
            PointerWidth::U64 => types::I64,
        };

        let layout = DataLayout::new(model);

        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_description: DataDescription::new(),
            module,
            int_type: types::I64,
            real_type: types::F64,
            real_ptr_type: ptr_type,
            int_ptr_type: ptr_type,
            triple,
            layout,
        }
    }

    fn declare_function(&mut self, name: &str) -> Result<FuncId> {
        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined.
        //
        // TODO: This may be an area where the API should be streamlined; should
        // we have a version of `declare_function` that automatically declares
        // the function?
        let id = self
            .module
            .declare_function(&name, Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| anyhow!(e.to_string()))?;

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.module
            .define_function(id, &mut self.ctx)
            .map_err(|e| anyhow!(e.to_string()))?;

        Ok(id)
    }

    pub fn compile_set_u0(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[self.real_ptr_type.into(), self.real_ptr_type.into()];
        let arg_names = &["data", "u0"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        for a in model.time_indep_defns() {
            codegen.jit_compile_tensor(a, None)?;
        }

        codegen.jit_compile_tensor(model.state(), Some(*codegen.variables.get("u0").unwrap()))?;

        // Emit the return instruction.
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();

        self.declare_function("u0")
    }

    pub fn compile_calc_out(
        &mut self,
        model: &DiscreteModel,
    ) -> Result<FuncId> {
        let arg_types = &[
            self.real_type.into(),
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
        ];
        let arg_names = &["t", "u", "data"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        codegen.jit_compile_tensor(model.out(), Some(*codegen.variables.get(model.out().name()).unwrap()))?;
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();

        self.declare_function("calc_out")

    }

    pub fn compile_calc_stop(
        &mut self,
        model: &DiscreteModel,
    ) -> Result<FuncId> {
        let arg_types = &[
            self.real_type.into(),
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
        ];
        let arg_names = &["t", "u", "data", "root"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        if let Some(stop) = model.stop() {
            codegen.jit_compile_tensor(stop, None)?;
        }
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("calc_stop")
    }

    pub fn compile_rhs(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_type.into(),
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
        ];
        let arg_names = &["t", "u", "data", "rr"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        // calculate time dependant definitions
        for tensor in model.time_dep_defns() {
            codegen.jit_compile_tensor(tensor, None)?;
        }

        // TODO: could split state dep defns into before and after F
        for a in model.state_dep_defns() {
            codegen.jit_compile_tensor(a, None)?;
        }

        // F
        let res = *codegen.variables.get("rr").unwrap();
        codegen.jit_compile_tensor(model.rhs(), Some(res))?;

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("rhs")
    }

    pub fn compile_mass(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_type.into(),
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
        ];
        let arg_names = &["t", "dudt", "data", "rr"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        // only put code in this function if we have a state_dot and lhs
        if model.state_dot().is_some() && model.lhs().is_some() {
            // calculate time dependant definitions
            for tensor in model.time_dep_defns() {
                codegen.jit_compile_tensor(tensor, None)?;
            }

            for a in model.dstate_dep_defns() {
                codegen.jit_compile_tensor(a, None)?;
            }

            // mass
            let lhs = model.lhs().unwrap();
            let res = codegen.variables.get("rr").unwrap();
            codegen.jit_compile_tensor(lhs, Some(*res))?;
        }

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("mass")
    }

    pub fn compile_get_dims(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.int_ptr_type.into(),
            self.int_ptr_type.into(),
            self.int_ptr_type.into(),
            self.int_ptr_type.into(),
            self.int_ptr_type.into(),
        ];
        let arg_names = &["states", "inputs", "outputs", "data", "stop"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let number_of_states = i64::try_from(model.state().nnz()).unwrap();
        let number_of_inputs = i64::try_from(model.inputs().iter().fold(0, |acc, x| acc + x.nnz())).unwrap();
        let number_of_outputs = i64::try_from(model.out().nnz()).unwrap();
        let number_of_stop = if let Some(stop) = model.stop() {
            i64::try_from(stop.nnz()).unwrap()
        } else {
            0
        };
        let data_len = i64::try_from(codegen.layout.data().len()).unwrap();

        for (val, name) in [
            (number_of_states, "states"),
            (number_of_inputs, "inputs"),
            (number_of_outputs, "outputs"),
            (data_len, "data"),
            (number_of_stop, "stop"),
        ] {
            let val = codegen.builder.ins().iconst(codegen.int_type, val);
            let ptr = codegen.variables.get(name).unwrap();
            let ptr = codegen.builder.use_var(*ptr);
            codegen.builder.ins().store(codegen.mem_flags, val, ptr, 0);
        }

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("gen_dims")
    }

    pub fn compile_get_tensor(
        &mut self,
        model: &DiscreteModel,
        name: &str,
    ) -> Result<FuncId> {
        let arg_types = &[
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
            self.int_ptr_type.into(),
        ];
        let arg_names = &["data", "tensor_data", "tensor_size"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);


        let tensor_ptr = codegen.variables.get(name).unwrap();
        let tensor_ptr = codegen.builder.use_var(*tensor_ptr);

        let tensor_size = i64::try_from(codegen.layout.get_layout(name).unwrap().nnz()).unwrap();
        let tensor_size = codegen.builder.ins().iconst(codegen.int_type, tensor_size);
    
        for (val, name) in [(tensor_ptr, "tensor_data"), (tensor_size, "tensor_size")] {
            let ptr = codegen.variables.get(name).unwrap();
            let ptr = codegen.builder.use_var(*ptr);
            codegen.builder.ins().store(codegen.mem_flags, val, ptr, 0);
        }

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("get_tensor")
    }

    pub fn compile_set_inputs(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_ptr_type.into(),
            self.real_ptr_type.into(),
        ];
        let arg_names = &["inputs", "data"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let mut inputs_index = 0usize;
        let base_data_ptr = codegen.variables.get("data").unwrap();
        let base_data_ptr = codegen.builder.use_var(*base_data_ptr);
        for input in model.inputs() {
            codegen.insert_tensor(input, base_data_ptr);
            let data_ptr = codegen.variables.get(input.name()).unwrap();
            let data_ptr = codegen.builder.use_var(*data_ptr);
            let input_ptr = codegen.variables.get("inputs").unwrap();
            let input_ptr = codegen.builder.use_var(*input_ptr);

            // loop thru the elements of this input and set them using the inputs ptr
            let inputs_start_index = codegen.builder.ins().iconst(codegen.int_type, i64::try_from(inputs_index).unwrap());
            let start_index = codegen.builder.ins().iconst(codegen.int_type, 0);

            let input_block = codegen.builder.create_block();
            let curr_input_index = codegen.builder.append_block_param(input_block, codegen.int_type);
            codegen.builder.ins().jump(input_block, &[start_index]);
            codegen.builder.switch_to_block(input_block);

            // loop body - copy value from inputs to data
            let indexed_input_ptr = codegen.builder.ins().iadd(input_ptr, curr_input_index);
            let indexed_data_ptr = codegen.builder.ins().iadd(data_ptr, curr_input_index);
            let input_value = codegen.builder.ins().load(codegen.real_type, codegen.mem_flags, indexed_input_ptr, 0);
            codegen.builder.ins().store(codegen.mem_flags, input_value, indexed_data_ptr, 0);


            // increment loop index
            let one = codegen.builder.ins().iconst(codegen.int_type, 1);
            let next_index = codegen.builder.ins().iadd(curr_input_index, one);

            let loop_while = codegen.builder.ins().icmp_imm(IntCC::UnsignedLessThan, next_index, i64::try_from(input.nnz()).unwrap());
            let post_block = codegen.builder.create_block();
            codegen.builder.ins().brif(
                loop_while,
                input_block,
                &[next_index],
                post_block,
                &[],
            );
            codegen.builder.seal_block(input_block);
            codegen.builder.seal_block(post_block);
            codegen.builder.switch_to_block(post_block);

            // get ready for next input
            inputs_index += input.nnz();
        }

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("set_inputs")
    }

    pub fn compile_set_id(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[self.real_ptr_type.into()];
        let arg_names = &["id"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let mut id_index = 0usize;
        for (blk, is_algebraic) in zip(model.state().elmts(), model.is_algebraic()) {
            // loop thru the elements of this state blk and set the corresponding elements of id
            let id_start_index = codegen.builder.ins().iconst(codegen.int_type, i64::try_from(id_index).unwrap());
            let blk_start_index = codegen.builder.ins().iconst(codegen.int_type, 0);
            
            let blk_block = codegen.builder.create_block();
            let curr_blk_index = codegen.builder.append_block_param(blk_block, codegen.int_type);
            codegen.builder.ins().jump(blk_block, &[blk_start_index]);

            codegen.builder.switch_to_block(blk_block);

            // loop body - copy value from inputs to data
            let input_id_ptr = codegen.variables.get("id").unwrap();
            let input_id_ptr = codegen.builder.use_var(*input_id_ptr);
            let curr_id_index= codegen.builder.ins().iadd(id_start_index, curr_blk_index);
            let indexed_id_ptr = codegen.builder.ins().iadd(input_id_ptr, curr_id_index);


            let is_algebraic_float = if *is_algebraic {
                0.0
            } else {
                1.0
            };
            let is_algebraic_value = codegen.fconst(is_algebraic_float);
            codegen.builder.ins().store(codegen.mem_flags, indexed_id_ptr, is_algebraic_value, 0);

            // increment loop index
            let one = codegen.builder.ins().iconst(codegen.int_type, 1);
            let next_index = codegen.builder.ins().iadd(curr_blk_index, one);

            let loop_while = codegen.builder.ins().icmp_imm(IntCC::UnsignedLessThan, next_index, i64::try_from(blk.nnz()).unwrap());
            let post_block = codegen.builder.create_block();
            codegen.builder.ins().brif(
                loop_while,
                blk_block,
                &[next_index],
                post_block,
                &[],
            );
            codegen.builder.seal_block(blk_block);
            codegen.builder.seal_block(post_block);
            codegen.builder.switch_to_block(post_block);

            // get ready for next blk
            id_index += blk.nnz();
        }

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("set_id")
    }

}



/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct CraneliftCodeGen<'a> {
    int_type: types::Type,
    real_type: types::Type,
    real_ptr_type: types::Type,
    int_ptr_type: types::Type,
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    tensor_ptr: Option<Value>,
    variables: HashMap<String, Variable>,
    mem_flags: MemFlags,
    functions: HashMap<String, FuncRef>,
    layout: &'a DataLayout,
}


impl<'ctx> CraneliftCodeGen<'ctx> {
    
    fn fconst(&mut self, value: f64) -> Value {
        match self.real_type {
            types::F32 => self.builder.ins().f32const(value as f32),
            types::F64 => self.builder.ins().f64const(value),
            _ => panic!("unexpected real type"),
        }
    }
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
            AstKind::Number(value) => Ok(self.fconst(*value)),
            AstKind::IndexedName(iname) => {
                let ptr = self.builder.use_var(*self.variables.get(name).unwrap());
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
                            iname_elmt_index =
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
                    Some(offset) => self.builder.ins().iadd(ptr, offset),
                    None => ptr,
                };
                Ok(self.builder.ins().load(self.real_type, self.mem_flags, value_ptr, 0))
            }
            AstKind::Name(name) => {
                // must be a scalar, just load the value
                let ptr = self.builder.use_var(*self.variables.get(*name).unwrap());
                Ok(self.builder.ins().load(self.real_type, self.mem_flags, ptr, 0))
            }
            AstKind::NamedGradient(name) => {
                let name_str = name.to_string();
                let ptr = self.builder.use_var(*self.variables.get(name_str.as_str()).unwrap());
                Ok(self.builder.ins().load(self.real_type, self.mem_flags, ptr, 0))
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
        var: Option<Variable>,
    ) -> Result<Value> {
        // set up the tensor storage pointer and index into this data
        if let Some(var) = var {
            self.tensor_ptr = Some(self.builder.use_var(var));
        } else {
            let res_ptr_var = *self.variables.get(a.name()).expect(format!("tensor {} not defined", a.name()).as_str());
            let res_ptr = self.builder.use_var(res_ptr_var);
            self.tensor_ptr = Some(res_ptr);
        }

        // treat scalar as a special case
        if a.rank() == 0 {

            let elmt = a.elmts().first().unwrap();
            let float_value = self.jit_compile_expr(a.name(), elmt.expr(), &[], elmt, None)?;
            self.builder.ins().store(self.mem_flags, float_value, self.tensor_ptr.unwrap(), 0);
        }

        for (i, blk) in a.elmts().iter().enumerate() {
            let default = format!("{}-{}", a.name(), i);
            let name = blk.name().unwrap_or(default.as_str());
            self.jit_compile_block(name, a, blk)?;
        }
        Ok(self.tensor_ptr.unwrap())
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
        self.builder.seal_block(preblock);
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
                let fzero = self.fconst(0.0);
                self.builder.ins().stack_store(fzero, contract_sum.unwrap(), 0);
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
            self.jit_compile_broadcast_and_store(
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
            let loop_cond = self.builder.ins().icmp_imm(IntCC::UnsignedLessThan, next_index, expr_shape[i]);
            self.builder.ins().brif(
                loop_cond,
                blocks[i],
                &[next_index],
                block,
                &[],
            );
            self.builder.seal_block(blocks[i]);
            self.builder.seal_block(block);
            self.builder.switch_to_block(block);
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
        let zero = self.builder.ins().iconst(int_type, 0);
        let one = self.builder.ins().iconst(int_type, 1);
        let two = self.builder.ins().iconst(int_type, 2);


        let layout_index = self.layout.get_layout_index(elmt.expr_layout()).unwrap();
        let translation_index = self
            .layout
            .get_translation_index(elmt.expr_layout(), elmt.layout())
            .unwrap();
        let translation_index = translation_index + translation.get_from_index_in_data_layout();

        // initialise the contract sum
        let contract_sum_var = self.decl_stack_slot(self.real_type, None);

        // loop through each contraction
        let block = self.builder.create_block();
        let contract_index = self.builder.append_block_param(block, self.int_type);
        let initial_contract_index = zero;
        let final_contract_index = self.builder.ins().iconst(
            int_type,
            i64::try_from(elmt.layout().nnz()).unwrap(),
        );
        self.builder.ins().jump(block, &[initial_contract_index]);
        self.builder.switch_to_block(block);


        // start and end indices stored next to each other in the indices array
        // start_index = translation_index + 2 * contract_index
        let translation_index_val = self.builder.ins().iconst(int_type, i64::try_from(translation_index).unwrap());
        let double_contract_index = self.builder.ins().imul(
            two,
            contract_index,
        );
        let start_index = self.builder.ins().iadd(
            translation_index_val,
           double_contract_index 
        );
        // end_index = start_index + 1
        let end_index = self.builder.ins().iadd(start_index, one);

        // index into the indices array to get the start and end indices
        // start_contract = indices[translation_index + 2 * contract_index]
        // end_contract = indices[translation_index + 2 * contract_index + 1]
        let indices_array = self.builder.use_var(*self.variables.get("indices").unwrap());
        let ptr = self.builder.ins().iadd(indices_array, start_index);
        let start_contract = self.builder.ins().load(self.int_type, self.mem_flags, ptr, 0);
        let ptr = self.builder.ins().iadd(indices_array, end_index);
        let end_contract = self.builder.ins().load(self.int_type, self.mem_flags, ptr, 0);
        

        // loop through each element in the contraction
        let contract_block = self.builder.create_block();
        let expr_index = self.builder.append_block_param(contract_block, self.int_type);
        self.builder.ins().jump(block, &[start_contract]);
        self.builder.switch_to_block(block);

        // init sum
        let fzero = self.fconst(0.0);
        self.builder.ins().stack_store(fzero, contract_sum_var, 0);

        // loop body - load index from layout
        let rank_val = self.builder.ins().iconst(self.int_type, i64::try_from(elmt.expr_layout().rank()).unwrap());
        let elmt_index_mult_rank = self.builder.ins().imul(
            expr_index,
            rank_val
        );
        let indices_int = (0..elmt.expr_layout().rank())
            // index = indices[layout_index + i + elmt_index * rank]
            .map(|i| {
                let layout_index_plus_offset = self.builder.ins().iconst(self.int_type, i64::try_from(layout_index + i).unwrap());
                let curr_index = self.builder.ins().iadd(
                    elmt_index_mult_rank,
                    layout_index_plus_offset
                );
                let ptr = self.builder.ins().iadd(indices_array, curr_index);
                let index = self.builder.ins().load(self.int_type, self.mem_flags, ptr, 0);
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
        let contract_sum_value = self.builder.ins().stack_load(self.real_type, contract_sum_var, 0);
        let new_contract_sum_value = self.builder.ins().fadd(contract_sum_value, float_value);
        self.builder.ins().stack_store(new_contract_sum_value, contract_sum_var, 0);

        // increment contract loop index
        let next_elmt_index = self.builder.ins().iadd(expr_index, one);

        // contract loop condition
        let loop_while = self.builder.ins().icmp(IntCC::UnsignedLessThan, next_elmt_index, end_contract);
        let post_contract_block = self.builder.create_block();
        self.builder.ins().brif(loop_while, contract_block, &[next_elmt_index], post_contract_block, &[]);
        self.builder.seal_block(contract_block);
        self.builder.seal_block(post_contract_block);
        
        self.builder.switch_to_block(post_contract_block);

        // store the result
        self.jit_compile_store(
            name,
            elmt,
            contract_index,
            new_contract_sum_value,
            translation,
        )?;

        // increment outer loop index
        let next_contract_index = self.builder.ins().iadd(
            contract_index, one
        );


        // outer loop condition
        let loop_while = self.builder.ins().icmp(IntCC::UnsignedLessThan, next_contract_index, final_contract_index);
        let post_block = self.builder.create_block();
        self.builder.ins().brif(loop_while, block, &[next_contract_index], post_block, &[]);
        self.builder.seal_block(block);
        self.builder.switch_to_block(post_block);
        self.builder.seal_block(post_block);

        Ok(())
    }

    // for sparse blocks we can loop through the non-zero elements and extract the index from the layout, then we compile the expression passing in this index
    // TODO: havn't implemented contractions yet
    fn jit_compile_sparse_block(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        translation: &Translation,
    ) -> Result<()> {
        let int_type = self.int_type;

        let layout_index = self.layout.get_layout_index(elmt.expr_layout()).unwrap();

        // loop through the non-zero elements
        let zero = self.builder.ins().iconst(int_type, 0);
        let one = self.builder.ins().iconst(int_type, 1);
        let start_index = zero;
        let end_index = self.builder.ins().iconst(
            int_type,
            i64::try_from(elmt.layout().nnz()).unwrap(),
        );

        let mut block = self.builder.create_block();
        let curr_index = self.builder.append_block_param(block, int_type);
        self.builder.ins().jump(block, &[start_index]);
        self.builder.switch_to_block(block);

        // loop body - load index from layout
        let elmt_index = curr_index;
        let rank_val = self.builder.ins().iconst(int_type, i64::try_from(elmt.expr_layout().rank()).unwrap());
        let elmt_index_mult_rank = self.builder.ins().imul(
            elmt_index,
            rank_val
        );
        let indices_int = (0..elmt.expr_layout().rank())
            // index = indices[layout_index + i + elmt_index * rank]
            .map(|i| {
                let layout_index_plus_offset = self.builder.ins().iconst(int_type, i64::try_from(layout_index + i).unwrap());
                let curr_index = self.builder.ins().iadd(
                    elmt_index_mult_rank,
                    layout_index_plus_offset,
                );
                let indices_ptr = self.builder.use_var(*self.variables.get("indices").unwrap());
                let ptr = self.builder.ins().iadd(indices_ptr, curr_index);
                let index = self.builder.ins().load(self.int_type, self.mem_flags, ptr, 0);
                Ok(index)
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        // loop body - eval expression
        let float_value = self.jit_compile_expr(
            name,
            elmt.expr(),
            indices_int.as_slice(),
            elmt,
            Some(elmt_index),
        )?;

        block = self.jit_compile_broadcast_and_store(
            name,
            elmt,
            float_value,
            elmt_index,
            translation,
            block,
        )?;

        // increment loop index
        let next_index = self.builder.ins().iadd(elmt_index, one);

        // loop condition
        let loop_while = self.builder.ins().icmp(IntCC::UnsignedLessThan, next_index, end_index);
        let post_block = self.builder.create_block();

        self.builder.ins().brif(loop_while, block, &[next_index], post_block, &[]);
        self.builder.seal_block(block);
        self.builder.switch_to_block(post_block);
        self.builder.seal_block(post_block);
        Ok(())
    }

     // for diagonal blocks we can loop through the diagonal elements and the index is just the same for each element, then we compile the expression passing in this index
    fn jit_compile_diagonal_block(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        translation: &Translation,
    ) -> Result<()> {
        let int_type = self.int_type;

        // loop through the non-zero elements
        let zero = self.builder.ins().iconst(int_type, 0);
        let one = self.builder.ins().iconst(int_type, 1);
        let mut block = self.builder.create_block();
        let start_index = zero;
        let end_index = self.builder.ins().iconst(int_type, i64::try_from(elmt.expr_layout().nnz()).unwrap());
        let curr_index = self.builder.append_block_param(block, int_type);
        self.builder.ins().jump(block, &[start_index]);
        self.builder.switch_to_block(block);

        // loop body - index is just the same for each element
        let elmt_index = curr_index;
        let indices_int = vec![elmt_index; elmt.expr_layout().rank()];

        // loop body - eval expression
        let float_value = self.jit_compile_expr(
            name,
            elmt.expr(),
            indices_int.as_slice(),
            elmt,
            Some(elmt_index),
        )?;

        // loop body - store result
        block = self.jit_compile_broadcast_and_store(
            name,
            elmt,
            float_value,
            elmt_index,
            translation,
            block,
        )?;

        // increment loop index
        let next_index = self.builder.ins().iadd(elmt_index, one);
        let loop_while = self.builder.ins().icmp(IntCC::UnsignedLessThan, next_index, end_index);
        let post_block = self.builder.create_block();
        self.builder.ins().brif(loop_while, block, &[next_index], post_block, &[]);
        self.builder.seal_block(block);
        self.builder.switch_to_block(post_block);
        self.builder.seal_block(post_block);

        Ok(())
    }

     fn jit_compile_broadcast_and_store(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        float_value: Value,
        expr_index: Value,
        translation: &Translation,
        pre_block: Block,
    ) -> Result<Block> {
        let int_type = self.int_type;
        let one = self.builder.ins().iconst(int_type, 1);
        let zero = self.builder.ins().iconst(int_type, 0);
        match translation.source {
            TranslationFrom::Broadcast {
                broadcast_by: _,
                broadcast_len,
            } => {
                let bcast_block = self.builder.create_block();
                let bcast_start_index = zero;
                let bcast_end_index = self.builder.ins().iconst(int_type, i64::try_from(broadcast_len).unwrap());
                let bcast_index = self.builder.append_block_param(bcast_block, self.int_type);

                // setup loop block
                self.builder.ins().jump(bcast_block, &[bcast_start_index]);
                self.builder.seal_block(bcast_block);
                self.builder.switch_to_block(bcast_block);


                // store value at index = expr_index * broadcast_len + bcast_index
                let tmp = self.builder.ins().imul(expr_index, bcast_end_index);
                let store_index = self.builder.ins().iadd(
                    tmp,
                    bcast_index,
                );
                self.jit_compile_store(name, elmt, store_index, float_value, translation)?;

                // increment index
                let bcast_next_index = self.builder.ins().iadd(
                    bcast_index,
                    one,
                );
                let bcast_cond = self.builder.ins().icmp(IntCC::UnsignedLessThan, bcast_next_index, bcast_end_index);
                let post_bcast_block = self.builder.create_block();
                self.builder.ins().brif(bcast_cond, bcast_block, &[bcast_next_index], post_bcast_block, &[]);
                self.builder.seal_block(bcast_block);
                self.builder.switch_to_block(post_bcast_block);

                // return the current block for later
                Ok(post_bcast_block)
            }
            TranslationFrom::ElementWise | TranslationFrom::DiagonalContraction { .. } => {
                self.jit_compile_store(name, elmt, expr_index, float_value, translation)?;
                Ok(pre_block)
            }
            _ => Err(anyhow!("Invalid translation")),
        }
    }

    fn jit_compile_store(
        &mut self,
        _name: &str,
        elmt: &TensorBlock,
        store_index: Value,
        float_value: Value,
        translation: &Translation,
    ) -> Result<()> {
        let int_type = self.int_type;
        let rank = elmt.layout().rank();
        let res_index = match &translation.target {
            TranslationTo::Contiguous { start, end: _ } => {
                let start_const = self.builder.ins().iconst(int_type, i64::try_from(*start).unwrap());
                self.builder.ins().iadd(start_const, store_index)
            }
            TranslationTo::Sparse { indices: _ } => {
                // load store index from layout
                let translate_index = self
                    .layout
                    .get_translation_index(elmt.expr_layout(), elmt.layout())
                    .unwrap();
                let translate_store_index =
                    translate_index + translation.get_to_index_in_data_layout();
                let translate_store_index = self.builder.ins().iconst(int_type, i64::try_from(translate_store_index).unwrap());
                let rank_const = self.builder.ins().iconst(int_type, i64::try_from(rank).unwrap());
                let elmt_index_strided = self.builder.ins().imul(store_index, rank_const);
                let curr_index = self.builder.ins().iadd(elmt_index_strided, translate_store_index);
                let indices_ptr = self.builder.use_var(*self.variables.get("indices").unwrap());
                let ptr = self.builder.ins().iadd(indices_ptr, curr_index);
                self.builder.ins().load(self.int_type, self.mem_flags, ptr, 0)
            }
        };

        let ptr = self.builder.ins().iadd(*self.tensor_ptr.as_ref().unwrap(), res_index);
        self.builder.ins().store(
            self.mem_flags,
            float_value,
            ptr,
            0,
        );

        Ok(())
    }


    fn declare_variable(
        &mut self,
        ty: types::Type,
        name: &str,
        val: Value,
    ) -> Variable {
        let index = self.variables.len();
        let var = Variable::new(index);
        if !self.variables.contains_key(name) {
            self.variables.insert(name.into(), var);
            self.builder.declare_var(var, ty);
            self.builder.def_var(var, val);
        }
        var
    }

    fn insert_tensor(&mut self, tensor: &Tensor, ptr: Value) {
        let mut tensor_data_index = i64::try_from(self.layout.get_data_index(tensor.name()).unwrap()).unwrap();
        let tensor_data_index_val = self.builder.ins().iconst(self.int_type, tensor_data_index);
        let tensor_data_ptr = self.builder.ins().iadd(ptr, tensor_data_index_val);
        self.declare_variable(self.real_ptr_type, tensor.name(), tensor_data_ptr);
        
        //insert any named blocks
        for blk in tensor.elmts() {
            if let Some(name) = blk.name() {
                let tensor_data_index_val = self.builder.ins().iconst(self.int_type, tensor_data_index);
                let tensor_data_ptr = self.builder.ins().iadd(ptr, tensor_data_index_val);
                self.declare_variable(self.real_ptr_type, name, tensor_data_ptr);
            }
            // named blocks only supported for rank <= 1, so we can just add the nnz to get the next data index
            tensor_data_index += i64::try_from(blk.nnz()).unwrap();
        }
    }

    pub fn new(module: &'ctx mut CraneliftModule, model: &DiscreteModel, arg_names: &[&str], arg_types: &[Type]) -> Self {
        module.ctx.func.signature.params.clear();
        module.ctx.func.signature.returns.clear();

        for ty in arg_types {
            module.ctx.func.signature.params.push(AbiParam::new(*ty));
        }

        // Create the builder to build a function.
        let mut builder = FunctionBuilder::new(&mut module.ctx.func, &mut module.builder_context);

        // Create the entry block, to start emitting code in.
        let entry_block = builder.create_block();

        // Since this is the entry block, add block parameters corresponding to
        // the function's parameters.
        //
        // TODO: Streamline the API here.
        builder.append_block_params_for_function_params(entry_block);

        // Tell the builder to emit code in this block.
        builder.switch_to_block(entry_block);

        // And, tell the builder that this block will have no further
        // predecessors. Since it's the entry block, it won't have any
        // predecessors.
        builder.seal_block(entry_block);

        let mut codegen = Self {
            int_type: module.int_type,
            real_type: module.real_type,
            real_ptr_type: module.real_ptr_type,
            int_ptr_type: module.int_ptr_type,
            builder,
            module: &mut module.module,
            tensor_ptr: None,
            variables: HashMap::new(),
            mem_flags: MemFlags::new(),
            functions: HashMap::new(),
            layout: &module.layout,
        };

        // insert arg vars
        for (i, (arg_name, arg_type)) in arg_names.iter().zip(arg_types.iter()).enumerate() {
            let val = codegen.builder.block_params(entry_block)[i];
            codegen.declare_variable(*arg_type,  *arg_name, val);
        }

        // insert u if it exists in args
        if let Some(u) = codegen.variables.get("u") {
            let u_ptr = codegen.builder.use_var(*u);
            codegen.insert_tensor(model.state(), u_ptr);
        }

        // insert dudt if it exists in args and is used in the model
        if let Some(dudt) = codegen.variables.get("dudt") {
            if let Some(state_dot) = model.state_dot() {
                let statedot_ptr = codegen.builder.use_var(*dudt);
                codegen.insert_tensor(state_dot, statedot_ptr);
            }
        }

        // insert all tensors in data if it exists in args
        if let Some(data) = codegen.variables.get("data") {
            let tensors = model.inputs().iter();
            let tensors = tensors.chain(model.time_indep_defns().iter());
            let tensors = tensors.chain(model.time_dep_defns().iter());
            let tensors = tensors.chain(model.state_dep_defns().iter());
            let mut others = Vec::new();
            others.push(model.out());
            others.push(model.rhs());
            if let Some(lhs) = model.lhs() {
                others.push(lhs);
            }
            let tensors = tensors.chain(others.into_iter());

            let data_ptr = codegen.builder.use_var(*data);

            for tensor in tensors {
                codegen.insert_tensor(tensor, data_ptr);
            }
        }
        codegen
    }
}




