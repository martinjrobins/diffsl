use anyhow::{anyhow, Ok, Result};
use codegen::ir::{AtomicRmwOp, FuncRef, GlobalValue, StackSlot};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, FuncId, FuncOrDataId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use std::iter::zip;
use target_lexicon::{Endianness, PointerWidth, Triple};

use crate::ast::{Ast, AstKind};
use crate::discretise::{DiscreteModel, Tensor, TensorBlock};
use crate::execution::compiler::CompilerMode;
use crate::execution::module::{
    CodegenModule, CodegenModuleCompile, CodegenModuleEmit, CodegenModuleJit,
};
use crate::execution::{DataLayout, Translation, TranslationFrom, TranslationTo};

pub struct CraneliftModule<M: Module> {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The module, with the object backend.
    module: M,

    layout: DataLayout,

    indices_id: DataId,
    constants_id: DataId,
    thread_counter: Option<DataId>,

    //triple: Triple,
    int_type: types::Type,
    real_type: types::Type,
    real_ptr_type: types::Type,
    int_ptr_type: types::Type,
    threaded: bool,
}

pub type CraneliftJitModule = CraneliftModule<JITModule>;
pub type CraneliftObjectModule = CraneliftModule<ObjectModule>;

impl<M: Module> CraneliftModule<M> {
    fn declare_function(&mut self, name: &str) -> Result<FuncId> {
        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined.
        //
        // TODO: This may be an area where the API should be streamlined; should
        // we have a version of `declare_function` that automatically declares
        // the function?
        let id = self
            .module
            .declare_function(name, Linkage::Export, &self.ctx.func.signature)?;

        //println!("Declared function: {} -------------------------------------------------------------------------------------", name);
        //println!("IR:\n{}", self.ctx.func);

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.module.define_function(id, &mut self.ctx)?;

        // Now that compilation is finished, we can clear out the context state.
        self.module.clear_context(&mut self.ctx);

        Ok(id)
    }

    fn compile_barrier_init(&mut self) -> Result<FuncId> {
        let module = self;
        module.ctx.func.signature.params.clear();
        module.ctx.func.signature.returns.clear();

        // Create the builder to build a function.
        let mut builder = FunctionBuilder::new(&mut module.ctx.func, &mut module.builder_context);

        let thread_counter = module
            .module
            .declare_data_in_func(module.thread_counter.unwrap(), builder.func);

        let entry_block = builder.create_block();

        // Tell the builder to emit code in this block.
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // load the total thread count
        let thread_counter = builder
            .ins()
            .global_value(module.int_ptr_type, thread_counter);

        // zero the barrier counter
        let zero = builder.ins().iconst(module.int_type, 0);
        builder
            .ins()
            .store(MemFlags::new(), zero, thread_counter, 0);

        builder.ins().return_(&[]);
        builder.finalize();

        let name = "barrier_init";
        let id =
            module
                .module
                .declare_function(name, Linkage::Export, &module.ctx.func.signature)?;
        //println!("Declared function: {} -------------------------------------------------------------------------------------", name);
        //println!("IR:\n{}", module.ctx.func);

        module.module.define_function(id, &mut module.ctx)?;
        module.module.clear_context(&mut module.ctx);

        Ok(id)
    }

    fn compile_barrier(&mut self) -> Result<FuncId> {
        let module = self;
        module.ctx.func.signature.params.clear();
        module.ctx.func.signature.returns.clear();

        // arg is the number of threads
        let arg_types = &[module.int_type];
        for ty in arg_types {
            module.ctx.func.signature.params.push(AbiParam::new(*ty));
        }

        // Create the builder to build a function.
        let mut builder = FunctionBuilder::new(&mut module.ctx.func, &mut module.builder_context);

        let thread_counter = module
            .module
            .declare_data_in_func(module.thread_counter.unwrap(), builder.func);

        let entry_block = builder.create_block();
        let wait_loop_block = builder.create_block();
        let barrier_done_block = builder.create_block();

        builder.append_block_params_for_function_params(entry_block);
        let thread_count = builder.block_params(entry_block)[0];

        // Tell the builder to emit code in this block.
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // load the total thread count
        let thread_counter = builder
            .ins()
            .global_value(module.int_ptr_type, thread_counter);

        // Atomically increment the barrier counter
        let one = builder.ins().iconst(module.int_type, 1);
        builder.ins().atomic_rmw(
            module.int_type,
            MemFlags::new(),
            AtomicRmwOp::Add,
            thread_counter,
            one,
        );

        // wait_loop:
        builder.ins().jump(wait_loop_block, &[]);
        builder.switch_to_block(wait_loop_block);

        let current_value =
            builder
                .ins()
                .atomic_load(module.int_type, MemFlags::new(), thread_counter);

        let all_threads_done = builder.ins().icmp(
            IntCC::UnsignedGreaterThanOrEqual,
            current_value,
            thread_count,
        );

        builder.ins().brif(
            all_threads_done,
            barrier_done_block,
            &[],
            wait_loop_block,
            &[],
        );
        builder.seal_block(wait_loop_block);
        builder.switch_to_block(barrier_done_block);
        builder.seal_block(barrier_done_block);

        builder.ins().return_(&[]);
        builder.finalize();

        let name = "barrier";
        let id =
            module
                .module
                .declare_function(name, Linkage::Export, &module.ctx.func.signature)?;
        //println!("Declared function: {} -------------------------------------------------------------------------------------", name);
        //println!("IR:\n{}", module.ctx.func);

        module.module.define_function(id, &mut module.ctx)?;
        module.module.clear_context(&mut module.ctx);

        Ok(id)
    }
    fn compile_calc_out_grad(
        &mut self,
        _func_id: &FuncId,
        model: &DiscreteModel,
    ) -> Result<FuncId> {
        let arg_types = &[
            self.real_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &[
            "t",
            "u",
            "du",
            "data",
            "ddata",
            "out",
            "dout",
            "threadId",
            "threadDim",
        ];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        if let Some(out) = model.out() {
            codegen.jit_compile_tensor(out, None, true)?;
        }
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();

        self.declare_function("calc_out_grad")
    }

    fn compile_rhs_grad(&mut self, _func_id: &FuncId, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &[
            "t",
            "u",
            "du",
            "data",
            "ddata",
            "rr",
            "drr",
            "threadId",
            "threadDim",
        ];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        // calculate time dependant definitions
        let mut nbarrier = 0;
        for tensor in model.time_dep_defns() {
            codegen.jit_compile_tensor(tensor, None, true)?;
            codegen.jit_compile_call_barrier(nbarrier);
            nbarrier += 1;
        }

        // TODO: could split state dep defns into before and after F
        for a in model.state_dep_defns() {
            codegen.jit_compile_tensor(a, None, true)?;
            codegen.jit_compile_call_barrier(nbarrier);
            nbarrier += 1;
        }

        // F
        let res = *codegen.variables.get("drr").unwrap();
        codegen.jit_compile_tensor(model.rhs(), Some(res), true)?;

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("rhs_grad")
    }

    fn compile_set_inputs_grad(
        &mut self,
        _func_id: &FuncId,
        model: &DiscreteModel,
    ) -> Result<FuncId> {
        let arg_types = &[
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
        ];
        let arg_names = &["inputs", "dinputs", "data", "ddata"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let base_data_ptr = codegen.variables.get("ddata").unwrap();
        let base_data_ptr = codegen.builder.use_var(*base_data_ptr);
        codegen.jit_compile_inputs(model, base_data_ptr, true, false);

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("set_inputs_grad")
    }

    fn compile_set_constants(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[self.int_type, self.int_type];
        let arg_names = &["threadId", "threadDim"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let mut nbarrier = 0;
        #[allow(clippy::explicit_counter_loop)]
        for a in model.constant_defns() {
            codegen.jit_compile_tensor(a, None, false)?;
            codegen.jit_compile_call_barrier(nbarrier);
            nbarrier += 1;
        }
        // Emit the return instruction.
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();

        self.declare_function("set_constants")
    }

    fn compile_set_u0_grad(&mut self, _func_id: &FuncId, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &["u0", "du0", "data", "ddata", "threadId", "threadDim"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let mut nbarrier = 0;
        #[allow(clippy::explicit_counter_loop)]
        for a in model.input_dep_defns() {
            codegen.jit_compile_tensor(a, None, true)?;
            codegen.jit_compile_call_barrier(nbarrier);
            nbarrier += 1;
        }

        codegen.jit_compile_tensor(
            model.state(),
            Some(*codegen.variables.get("du0").unwrap()),
            true,
        )?;

        // Emit the return instruction.
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();

        self.declare_function("set_u0_grad")
    }

    fn new(triple: Triple, model: &DiscreteModel, threaded: bool, mut module: M) -> Result<Self> {
        let ptr_type = match triple.pointer_width().unwrap() {
            PointerWidth::U16 => types::I16,
            PointerWidth::U32 => types::I32,
            PointerWidth::U64 => types::I64,
        };

        let layout = DataLayout::new(model);

        // define constant global data
        let int_type = types::I32;
        let real_type = types::F64;
        let mut data_description = DataDescription::new();
        data_description.define_zeroinit(layout.constants().len() * (real_type.bytes() as usize));
        let constants_id = module.declare_data("constants", Linkage::Local, true, false)?;
        module.define_data(constants_id, &data_description)?;

        // write indices data as a global data object
        // convect the indices to bytes
        //let int_type = ptr_type;
        let mut vec8: Vec<u8> = vec![];
        for elem in layout.indices() {
            // convert indices to i64
            if int_type == types::I64 {
                let elemi64 = i64::from(*elem);
                let conv = match triple.endianness().unwrap() {
                    Endianness::Little => elemi64.to_le_bytes(),
                    Endianness::Big => elemi64.to_be_bytes(),
                };
                vec8.extend(conv.into_iter());
            } else {
                let conv = match triple.endianness().unwrap() {
                    Endianness::Little => elem.to_le_bytes(),
                    Endianness::Big => elem.to_be_bytes(),
                };
                vec8.extend(conv.into_iter());
            };
        }

        // put the indices data into a DataDescription
        let mut data_description = DataDescription::new();
        data_description.define(vec8.into_boxed_slice());
        let indices_id = module.declare_data("indices", Linkage::Local, false, false)?;
        module.define_data(indices_id, &data_description)?;

        let mut thread_counter = None;
        if threaded {
            let mut data_description = DataDescription::new();
            data_description.define_zeroinit(int_type.bytes().try_into().unwrap());
            let the_thread_counter =
                module.declare_data("thread_counter", Linkage::Local, true, false)?;
            module.define_data(the_thread_counter, &data_description)?;
            thread_counter = Some(the_thread_counter);
        }

        let mut ret = Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
            indices_id,
            constants_id,
            int_type,
            real_type,
            real_ptr_type: ptr_type,
            int_ptr_type: ptr_type,
            layout,
            threaded,
            thread_counter,
        };
        if threaded {
            ret.compile_barrier_init()?;
            ret.compile_barrier()?;
        }

        let set_u0 = ret.compile_set_u0(model)?;
        let _calc_stop = ret.compile_calc_stop(model)?;
        let rhs = ret.compile_rhs(model)?;
        let _mass = ret.compile_mass(model)?;
        let calc_out = ret.compile_calc_out(model)?;
        let _set_id = ret.compile_set_id(model)?;
        let _get_dims = ret.compile_get_dims(model)?;
        let set_inputs = ret.compile_set_inputs(model)?;
        let _get_inputs = ret.compile_get_inputs(model)?;
        let _set_constants = ret.compile_set_constants(model)?;
        let tensor_info = ret
            .layout
            .tensors()
            .map(|(name, is_constant)| (name.to_string(), is_constant))
            .collect::<Vec<_>>();
        for (tensor, is_constant) in tensor_info {
            if is_constant {
                ret.compile_get_constant(model, tensor.as_str())?;
            } else {
                ret.compile_get_tensor(model, tensor.as_str())?;
            }
        }
        let _set_u0_grad = ret.compile_set_u0_grad(&set_u0, model)?;
        let _rhs_grad = ret.compile_rhs_grad(&rhs, model)?;
        let _calc_out_grad = ret.compile_calc_out_grad(&calc_out, model)?;
        let _set_inputs_grad = ret.compile_set_inputs_grad(&set_inputs, model)?;
        Ok(ret)
    }

    fn compile_set_u0(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &["u0", "data", "threadId", "threadDim"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let mut nbarrier = 0;
        #[allow(clippy::explicit_counter_loop)]
        for a in model.input_dep_defns() {
            codegen.jit_compile_tensor(a, None, false)?;
            codegen.jit_compile_call_barrier(nbarrier);
            nbarrier += 1;
        }

        codegen.jit_compile_tensor(
            model.state(),
            Some(*codegen.variables.get("u0").unwrap()),
            false,
        )?;

        // Emit the return instruction.
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();

        self.declare_function("set_u0")
    }

    fn compile_calc_out(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &["t", "u", "data", "out", "threadId", "threadDim"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        if let Some(out) = model.out() {
            // calculate time dependant definitions
            let mut nbarrier = 0;
            for tensor in model.time_dep_defns() {
                codegen.jit_compile_tensor(tensor, None, false)?;
                codegen.jit_compile_call_barrier(nbarrier);
                nbarrier += 1;
            }

            // calculate state dependant definitions
            for a in model.state_dep_defns() {
                codegen.jit_compile_tensor(a, None, false)?;
                codegen.jit_compile_call_barrier(nbarrier);
                nbarrier += 1;
            }

            codegen.jit_compile_tensor(out, None, false)?;
        }
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();

        self.declare_function("calc_out")
    }

    fn compile_calc_stop(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &["t", "u", "data", "root", "threadId", "threadDim"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        if let Some(stop) = model.stop() {
            // calculate time dependant definitions
            let mut nbarrier = 0;
            for tensor in model.time_dep_defns() {
                codegen.jit_compile_tensor(tensor, None, false)?;
                codegen.jit_compile_call_barrier(nbarrier);
                nbarrier += 1;
            }

            // calculate state dependant definitions
            for a in model.state_dep_defns() {
                codegen.jit_compile_tensor(a, None, false)?;
                codegen.jit_compile_call_barrier(nbarrier);
                nbarrier += 1;
            }

            let root = *codegen.variables.get("root").unwrap();
            codegen.jit_compile_tensor(stop, Some(root), false)?;
        }
        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("calc_stop")
    }

    fn compile_rhs(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &["t", "u", "data", "rr", "threadId", "threadDim"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        // calculate time dependant definitions
        let mut nbarrier = 0;
        for tensor in model.time_dep_defns() {
            codegen.jit_compile_tensor(tensor, None, false)?;
            codegen.jit_compile_call_barrier(nbarrier);
            nbarrier += 1;
        }

        // TODO: could split state dep defns into before and after F
        for a in model.state_dep_defns() {
            codegen.jit_compile_tensor(a, None, false)?;
            codegen.jit_compile_call_barrier(nbarrier);
            nbarrier += 1;
        }

        // F
        let res = *codegen.variables.get("rr").unwrap();
        codegen.jit_compile_tensor(model.rhs(), Some(res), false)?;

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("rhs")
    }

    fn compile_mass(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.real_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.real_ptr_type,
            self.int_type,
            self.int_type,
        ];
        let arg_names = &["t", "dudt", "data", "rr", "threadId", "threadDim"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        // only put code in this function if we have a state_dot and lhs
        if model.state_dot().is_some() && model.lhs().is_some() {
            // calculate time dependant definitions
            let mut nbarrier = 0;
            for tensor in model.time_dep_defns() {
                codegen.jit_compile_tensor(tensor, None, false)?;
                codegen.jit_compile_call_barrier(nbarrier);
                nbarrier += 1;
            }

            for a in model.dstate_dep_defns() {
                codegen.jit_compile_tensor(a, None, false)?;
                codegen.jit_compile_call_barrier(nbarrier);
                nbarrier += 1;
            }

            // mass
            let lhs = model.lhs().unwrap();
            let res = codegen.variables.get("rr").unwrap();
            codegen.jit_compile_tensor(lhs, Some(*res), false)?;
        }

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("mass")
    }

    fn compile_get_dims(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[
            self.int_ptr_type,
            self.int_ptr_type,
            self.int_ptr_type,
            self.int_ptr_type,
            self.int_ptr_type,
            self.int_ptr_type,
        ];
        let arg_names = &["states", "inputs", "outputs", "data", "stop", "has_mass"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let number_of_states = i64::try_from(model.state().nnz()).unwrap();
        let number_of_inputs =
            i64::try_from(model.inputs().iter().fold(0, |acc, x| acc + x.nnz())).unwrap();
        let number_of_outputs = match model.out() {
            Some(out) => i64::try_from(out.nnz()).unwrap(),
            None => 0,
        };
        let number_of_stop = if let Some(stop) = model.stop() {
            i64::try_from(stop.nnz()).unwrap()
        } else {
            0
        };
        let has_mass = match model.lhs().is_some() {
            true => 1,
            false => 0,
        };
        let data_len = i64::try_from(codegen.layout.data().len()).unwrap();

        for (val, name) in [
            (number_of_states, "states"),
            (number_of_inputs, "inputs"),
            (number_of_outputs, "outputs"),
            (data_len, "data"),
            (number_of_stop, "stop"),
            (has_mass, "has_mass"),
        ] {
            let val = codegen.builder.ins().iconst(codegen.int_type, val);
            let ptr = codegen.variables.get(name).unwrap();
            let ptr = codegen.builder.use_var(*ptr);
            codegen.builder.ins().store(codegen.mem_flags, val, ptr, 0);
        }

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("get_dims")
    }

    fn compile_get_tensor(&mut self, model: &DiscreteModel, name: &str) -> Result<FuncId> {
        let arg_types = &[self.real_ptr_type, self.real_ptr_type, self.int_ptr_type];
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
        self.declare_function(format!("get_tensor_{name}").as_str())
    }

    fn compile_get_constant(&mut self, model: &DiscreteModel, name: &str) -> Result<FuncId> {
        let arg_types = &[self.real_ptr_type, self.int_ptr_type];
        let arg_names = &["tensor_data", "tensor_size"];
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
        self.declare_function(format!("get_constant_{name}").as_str())
    }

    fn compile_set_inputs(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[self.real_ptr_type, self.real_ptr_type];
        let arg_names = &["inputs", "data"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let base_data_ptr = codegen.variables.get("data").unwrap();
        let base_data_ptr = codegen.builder.use_var(*base_data_ptr);
        codegen.jit_compile_inputs(model, base_data_ptr, false, false);

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("set_inputs")
    }

    fn compile_get_inputs(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[self.real_ptr_type, self.real_ptr_type];
        let arg_names = &["inputs", "data"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let base_data_ptr = codegen.variables.get("data").unwrap();
        let base_data_ptr = codegen.builder.use_var(*base_data_ptr);
        codegen.jit_compile_inputs(model, base_data_ptr, false, true);

        codegen.builder.ins().return_(&[]);
        codegen.builder.finalize();
        self.declare_function("get_inputs")
    }

    fn compile_set_id(&mut self, model: &DiscreteModel) -> Result<FuncId> {
        let arg_types = &[self.real_ptr_type];
        let arg_names = &["id"];
        let mut codegen = CraneliftCodeGen::new(self, model, arg_names, arg_types);

        let mut id_index = 0usize;
        for (blk, is_algebraic) in zip(model.state().elmts(), model.is_algebraic()) {
            // loop thru the elements of this state blk and set the corresponding elements of id
            let id_start_index = codegen
                .builder
                .ins()
                .iconst(codegen.int_type, i64::try_from(id_index).unwrap());
            let blk_start_index = codegen.builder.ins().iconst(codegen.int_type, 0);

            let blk_block = codegen.builder.create_block();
            let curr_blk_index = codegen
                .builder
                .append_block_param(blk_block, codegen.int_type);
            codegen.builder.ins().jump(blk_block, &[blk_start_index]);

            codegen.builder.switch_to_block(blk_block);

            // loop body - copy value from inputs to data
            let input_id_ptr = codegen.variables.get("id").unwrap();
            let input_id_ptr = codegen.builder.use_var(*input_id_ptr);
            let curr_id_index = codegen.builder.ins().iadd(id_start_index, curr_blk_index);
            let indexed_id_ptr =
                codegen.ptr_add_offset(codegen.real_type, input_id_ptr, curr_id_index);

            let is_algebraic_float = if *is_algebraic { 0.0 } else { 1.0 };
            let is_algebraic_value = codegen.fconst(is_algebraic_float);
            codegen
                .builder
                .ins()
                .store(codegen.mem_flags, is_algebraic_value, indexed_id_ptr, 0);

            // increment loop index
            let one = codegen.builder.ins().iconst(codegen.int_type, 1);
            let next_index = codegen.builder.ins().iadd(curr_blk_index, one);

            let loop_while = codegen.builder.ins().icmp_imm(
                IntCC::UnsignedLessThan,
                next_index,
                i64::try_from(blk.nnz()).unwrap(),
            );
            let post_block = codegen.builder.create_block();
            codegen
                .builder
                .ins()
                .brif(loop_while, blk_block, &[next_index], post_block, &[]);
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

unsafe impl<M: Module> Sync for CraneliftModule<M> {}

impl<M: Module> CodegenModule for CraneliftModule<M> {}

impl CodegenModuleCompile for CraneliftModule<ObjectModule> {
    fn from_discrete_model(
        model: &DiscreteModel,
        mode: CompilerMode,
        triple: Option<Triple>,
    ) -> Result<Self> {
        let thread_dim = mode.thread_dim(model.state().nnz());
        let threaded = thread_dim > 1;

        let triple = triple.unwrap_or(Triple::host());
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        flag_builder.set("opt_level", "speed").unwrap();
        let flags = settings::Flags::new(flag_builder);
        let isa = isa::lookup(triple.clone())?.finish(flags)?;
        let builder =
            ObjectBuilder::new(isa, "diffsol", cranelift_module::default_libcall_names())?;

        let module = ObjectModule::new(builder);

        Self::new(triple, model, threaded, module)
    }
}

impl CodegenModuleCompile for CraneliftModule<JITModule> {
    fn from_discrete_model(
        model: &DiscreteModel,
        mode: CompilerMode,
        triple: Option<Triple>,
    ) -> Result<Self> {
        let thread_dim = mode.thread_dim(model.state().nnz());
        let threaded = thread_dim > 1;

        let triple = triple.unwrap_or(Triple::host());
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        flag_builder.set("opt_level", "speed").unwrap();
        let flags = settings::Flags::new(flag_builder);
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {msg}");
        });
        let isa = isa_builder.finish(flags).unwrap();
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // add supported external rust functions
        for func in crate::execution::functions::FUNCTIONS.iter() {
            builder.symbol(func.0, func.1 as *const u8);
            builder.symbol(
                CraneliftCodeGen::<JITModule>::get_function_name(func.0, true),
                func.2 as *const u8,
            );
        }
        for func in crate::execution::functions::TWO_ARG_FUNCTIONS.iter() {
            builder.symbol(func.0, func.1 as *const u8);
            builder.symbol(
                CraneliftCodeGen::<JITModule>::get_function_name(func.0, true),
                func.2 as *const u8,
            );
        }

        let module = JITModule::new(builder);
        Self::new(triple, model, threaded, module)
    }
}

impl CodegenModuleEmit for CraneliftModule<ObjectModule> {
    fn to_object(self) -> Result<Vec<u8>> {
        self.module.finish().emit().map_err(|e| anyhow!(e))
    }
}

impl CodegenModuleJit for CraneliftModule<JITModule> {
    fn jit(&mut self) -> Result<HashMap<String, *const u8>> {
        let mut result = HashMap::new();
        self.module.finalize_definitions()?;
        for (func, decl) in self.module.declarations().get_functions() {
            if Linkage::Import == decl.linkage {
                continue;
            }
            let addr = self.module.get_finalized_function(func);
            result.insert(decl.name.as_ref().unwrap().clone(), addr);
        }
        Ok(result)
    }
}

/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct CraneliftCodeGen<'a, M: Module> {
    int_type: types::Type,
    real_type: types::Type,
    real_ptr_type: types::Type,
    int_ptr_type: types::Type,
    builder: FunctionBuilder<'a>,
    module: &'a mut M,
    tensor_ptr: Option<Value>,
    variables: HashMap<String, Variable>,
    mem_flags: MemFlags,
    functions: HashMap<String, FuncRef>,
    layout: &'a DataLayout,
    indices: GlobalValue,
    constants: GlobalValue,
    threaded: bool,
}

impl<'ctx, M: Module> CraneliftCodeGen<'ctx, M> {
    fn fconst(&mut self, value: f64) -> Value {
        match self.real_type {
            types::F32 => self.builder.ins().f32const(value as f32),
            types::F64 => self.builder.ins().f64const(value),
            _ => panic!("unexpected real type"),
        }
    }
    fn ptr_add_offset_i64(&mut self, elmt_ty: types::Type, ptr: Value, offset: i64) -> Value {
        // both ptr types are the same, so just use real_ptr_type
        let ptr_ty = self.real_ptr_type;
        let width = elmt_ty.bytes() as i64;
        let offset_bytes = self.builder.ins().iconst(ptr_ty, offset * width);
        self.builder.ins().iadd(ptr, offset_bytes)
    }

    fn ptr_add_offset(&mut self, elmt_ty: types::Type, ptr: Value, offset: Value) -> Value {
        let width = elmt_ty.bytes() as i64;
        // both ptr types are the same, so just use real_ptr_type
        let ptr_ty = self.real_ptr_type;

        let width_value = self.builder.ins().iconst(ptr_ty, width);
        let offset_ptr = if self.int_type != ptr_ty {
            self.builder.ins().sextend(ptr_ty, offset)
        } else {
            offset
        };
        let offset_bytes = self.builder.ins().imul(offset_ptr, width_value);
        self.builder.ins().iadd(ptr, offset_bytes)
    }
    fn jit_compile_call_barrier(&mut self, nbarrier: i64) {
        if !self.threaded {
            return;
        }
        let thread_dim = self.variables.get("threadDim").unwrap();
        let thread_dim = self.builder.use_var(*thread_dim);
        let nbarrier = self.builder.ins().iconst(self.int_type, nbarrier + 1);
        let thread_dim_mul_nbarrier = self.builder.ins().imul(thread_dim, nbarrier);
        let barrier = self.get_function("barrier", false).unwrap();
        self.builder.ins().call(barrier, &[thread_dim_mul_nbarrier]);
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
            AstKind::Call(call) => match self.get_function(call.fn_name, call.is_tangent) {
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
            AstKind::Name(iname) => {
                let ptr = if iname.is_tangent {
                    // tangent of a constant is zero
                    if self.layout.is_constant(iname.name) {
                        return Ok(self.fconst(0.0));
                    }
                    let name = self.get_tangent_tensor_name(iname.name);
                    self.builder
                        .use_var(*self.variables.get(name.as_str()).unwrap())
                } else {
                    self.builder
                        .use_var(*self.variables.get(iname.name).unwrap())
                };
                // arg t is a special case (not a ptr)
                if iname.name == "t" {
                    return Ok(ptr);
                }
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
                            let stride_intval = self
                                .builder
                                .ins()
                                .iconst(self.int_type, i64::try_from(stride).unwrap());
                            let stride_mul_i = self.builder.ins().imul(stride_intval, iname_i);
                            iname_elmt_index =
                                self.builder.ins().iadd(iname_elmt_index, stride_mul_i);
                        }
                        Some(iname_elmt_index)
                    } else {
                        None
                    }
                } else if layout.is_sparse() || layout.is_diagonal() {
                    // must have come from jit_compile_sparse_block, so we can just use the elmt_index
                    // must have come from jit_compile_diagonal_block, so we can just use the elmt_index
                    expr_index
                } else {
                    panic!("unexpected layout");
                };
                let value_ptr = match iname_elmt_index {
                    Some(offset) => self.ptr_add_offset(self.real_type, ptr, offset),
                    None => ptr,
                };
                Ok(self
                    .builder
                    .ins()
                    .load(self.real_type, self.mem_flags, value_ptr, 0))
            }
            AstKind::NamedGradient(name) => {
                let name_str = name.to_string();
                let ptr = self
                    .builder
                    .use_var(*self.variables.get(name_str.as_str()).unwrap());
                Ok(self
                    .builder
                    .ins()
                    .load(self.real_type, self.mem_flags, ptr, 0))
            }
            AstKind::Index(_) => todo!(),
            AstKind::Slice(_) => todo!(),
            AstKind::Integer(_) => todo!(),
            _ => panic!("unexprected astkind"),
        }
    }

    fn get_function_name(name: &str, is_tangent: bool) -> String {
        if is_tangent {
            format!("{name}__tangent__")
        } else {
            name.to_owned()
        }
    }

    fn get_function(&mut self, base_name: &str, is_tangent: bool) -> Option<FuncRef> {
        let name = Self::get_function_name(base_name, is_tangent);
        match self.functions.get(name.as_str()) {
            Some(&func) => Some(func),
            None => {
                match crate::execution::functions::function_num_args(base_name, is_tangent) {
                    Some(num_args) => {
                        let mut sig = self.module.make_signature();
                        for _ in 0..num_args {
                            sig.params.push(AbiParam::new(self.real_type));
                        }
                        sig.returns.push(AbiParam::new(self.real_type));
                        let callee = self
                            .module
                            .declare_function(name.as_str(), Linkage::Import, &sig)
                            .expect("problem declaring function");
                        let function = self.module.declare_func_in_func(callee, self.builder.func);
                        self.functions.insert(name, function);
                        Some(function)
                    }
                    None => {
                        // not one of the supported external functions, so must be internal
                        match self.module.get_name(name.as_str()) {
                            Some(FuncOrDataId::Func(func_id)) => {
                                let function =
                                    self.module.declare_func_in_func(func_id, self.builder.func);
                                self.functions.insert(name, function);
                                Some(function)
                            }
                            _ => None,
                        }
                    }
                }
            }
        }
    }

    fn jit_compile_tensor(
        &mut self,
        a: &Tensor,
        var: Option<Variable>,
        is_tangent: bool,
    ) -> Result<Value> {
        // set up the tensor storage pointer and index into this data
        if let Some(var) = var {
            self.tensor_ptr = Some(self.builder.use_var(var));
        } else {
            let name = if is_tangent {
                self.get_tangent_tensor_name(a.name())
            } else {
                a.name().to_owned()
            };
            let res_ptr_var = *self
                .variables
                .get(name.as_str())
                .unwrap_or_else(|| panic!("tensor {} not defined", a.name()));
            let res_ptr = self.builder.use_var(res_ptr_var);
            self.tensor_ptr = Some(res_ptr);
        }

        // treat scalar as a special case
        if a.rank() == 0 {
            // if threaded then only the first thread will evaluate the scalar
            let mut exit_block = None;
            if self.threaded {
                let thread_id = self.variables.get("threadId").unwrap();
                let thread_id = self.builder.use_var(*thread_id);
                let is_first_thread = self.builder.ins().icmp_imm(IntCC::Equal, thread_id, 0);
                exit_block = Some(self.builder.create_block());
                let next_block = self.builder.create_block();
                self.builder
                    .ins()
                    .brif(is_first_thread, next_block, &[], exit_block.unwrap(), &[]);
                self.builder.seal_block(next_block);
                self.builder.switch_to_block(next_block);
            }
            let elmt = a.elmts().first().unwrap();
            let expr = if is_tangent {
                elmt.tangent_expr()
            } else {
                elmt.expr()
            };
            let float_value = self.jit_compile_expr(a.name(), expr, &[], elmt, None)?;
            self.builder
                .ins()
                .store(self.mem_flags, float_value, self.tensor_ptr.unwrap(), 0);

            // if threaded then carry on to the exit block
            if let Some(exit_block) = exit_block {
                self.builder.ins().jump(exit_block, &[]);
                self.builder.seal_block(exit_block);
                self.builder.switch_to_block(exit_block);
            }
        } else {
            for (i, blk) in a.elmts().iter().enumerate() {
                let default = format!("{}-{}", a.name(), i);
                let name = blk.name().unwrap_or(default.as_str());
                self.jit_compile_block(name, a, blk, is_tangent)?;
            }
        }

        Ok(self.tensor_ptr.unwrap())
    }

    fn jit_compile_block(
        &mut self,
        name: &str,
        tensor: &Tensor,
        elmt: &TensorBlock,
        is_tangent: bool,
    ) -> Result<()> {
        let translation = Translation::new(
            elmt.expr_layout(),
            elmt.layout(),
            elmt.start(),
            tensor.layout_ptr(),
        );

        if elmt.expr_layout().is_dense() {
            self.jit_compile_dense_block(name, elmt, &translation, is_tangent)
        } else if elmt.expr_layout().is_diagonal() {
            self.jit_compile_diagonal_block(name, elmt, &translation, is_tangent)
        } else if elmt.expr_layout().is_sparse() {
            match translation.source {
                TranslationFrom::SparseContraction { .. } => {
                    self.jit_compile_sparse_contraction_block(name, elmt, &translation, is_tangent)
                }
                _ => self.jit_compile_sparse_block(name, elmt, &translation, is_tangent),
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

    fn jit_threading_limits(&mut self, size: Value) -> (Value, Value, Block) {
        let one = self.builder.ins().iconst(self.int_type, 1);
        let thread_id = self.variables.get("threadId").unwrap();
        let thread_id = self.builder.use_var(*thread_id);
        let thread_dim = self.variables.get("threadDim").unwrap();
        let thread_dim = self.builder.use_var(*thread_dim);

        // start index is i * size / thread_dim
        let i_times_size = self.builder.ins().imul(thread_id, size);
        let start = self.builder.ins().udiv(i_times_size, thread_dim);

        // if start index is equal or greater than size then we are done and can exit
        let done = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, start, size);
        let exit_block = self.builder.create_block();
        let next_block = self.builder.create_block();
        self.builder
            .ins()
            .brif(done, exit_block, &[], next_block, &[]);
        self.builder.seal_block(next_block);
        self.builder.switch_to_block(next_block);

        // the ending index for thread i is min((i+1) * size / thread_dim, size)
        let i_plus_one = self.builder.ins().iadd(thread_id, one);
        let i_plus_one_times_size = self.builder.ins().imul(i_plus_one, size);
        let end = self.builder.ins().udiv(i_plus_one_times_size, thread_dim);
        let end_less_than_size = self.builder.ins().icmp(IntCC::UnsignedLessThan, end, size);
        let end = self.builder.ins().select(end_less_than_size, end, size);

        (start, end, exit_block)
    }

    // for dense blocks we can loop through the nested loops to calculate the index, then we compile the expression passing in this index
    fn jit_compile_dense_block(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        translation: &Translation,
        is_tangent: bool,
    ) -> Result<()> {
        let int_type = self.int_type;

        let expr_rank = elmt.expr_layout().rank();
        let expr_shape = elmt
            .expr_layout()
            .shape()
            .mapv(|n| i64::try_from(n).unwrap());
        let one = self.builder.ins().iconst(int_type, 1);
        let zero = self.builder.ins().iconst(int_type, 0);
        let mut expr_strides = vec![1i64; expr_rank];
        if expr_rank > 0 {
            for i in (0..expr_rank - 1).rev() {
                expr_strides[i] = expr_strides[i + 1] * expr_shape[i + 1];
            }
        }
        let expr_strides = expr_strides
            .iter()
            .map(|&s| self.builder.ins().iconst(int_type, s))
            .collect::<Vec<_>>();

        // setup indices, loop through the nested loops
        let mut indices = Vec::new();
        let mut blocks = Vec::new();

        // allocate the contract sum if needed
        let (contract_sum, contract_by, contract_strides) =
            if let TranslationFrom::DenseContraction {
                contract_by,
                contract_len: _,
            } = translation.source
            {
                let contract_rank = expr_rank - contract_by;
                let mut contract_strides = vec![1i64; contract_rank];
                for i in (0..contract_rank - 1).rev() {
                    contract_strides[i] = contract_strides[i + 1] * expr_shape[i + 1];
                }
                let contract_strides = contract_strides
                    .iter()
                    .map(|&s| self.builder.ins().iconst(int_type, s))
                    .collect::<Vec<_>>();

                (
                    Some(self.decl_stack_slot(self.real_type, None)),
                    contract_by,
                    Some(contract_strides),
                )
            } else {
                (None, 0, None)
            };

        // we will thread the output loop, except if we are contracting to a scalar
        let (thread_start, thread_end, exit_block) = if self.threaded {
            let expr_shape0 = self
                .builder
                .ins()
                .iconst(int_type, *expr_shape.get(0).unwrap_or(&1));
            let (start, end, exit_block) = self.jit_threading_limits(expr_shape0);
            (Some(start), Some(end), Some(exit_block))
        } else {
            (None, None, None)
        };

        for i in 0..expr_rank {
            let block = self.builder.create_block();
            let curr_index = self.builder.append_block_param(block, self.int_type);
            let curr_index_start = if i == 0 && self.threaded {
                thread_start.unwrap()
            } else {
                zero
            };
            self.builder.ins().jump(block, &[curr_index_start]);
            self.builder.switch_to_block(block);

            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                let fzero = self.fconst(0.0);
                self.builder
                    .ins()
                    .stack_store(fzero, contract_sum.unwrap(), 0);
            }

            indices.push(curr_index);
            blocks.push(block);
        }

        // load and increment the expression index
        //let expr_index = self
        //    .builder
        //    .ins()
        //    .stack_load(self.int_type, expr_index_var, 0);
        //let next_expr_index = self.builder.ins().iadd(expr_index, one);
        //self.builder
        //    .ins()
        //    .stack_store(next_expr_index, expr_index_var, 0);

        let expr = if is_tangent {
            elmt.tangent_expr()
        } else {
            elmt.expr()
        };
        let float_value = self.jit_compile_expr(name, expr, indices.as_slice(), elmt, None)?;

        if contract_sum.is_some() {
            let contract_sum_value =
                self.builder
                    .ins()
                    .stack_load(self.real_type, contract_sum.unwrap(), 0);
            let new_contract_sum_value = self.builder.ins().fadd(contract_sum_value, float_value);
            self.builder
                .ins()
                .stack_store(new_contract_sum_value, contract_sum.unwrap(), 0);
        } else {
            let expr_index = indices
                .iter()
                .zip(expr_strides.iter())
                .fold(zero, |acc, (i, s)| {
                    let tmp = self.builder.ins().imul(*i, *s);
                    self.builder.ins().iadd(acc, tmp)
                });
            self.jit_compile_broadcast_and_store(
                name,
                elmt,
                float_value,
                expr_index,
                translation,
                self.builder.current_block().unwrap(),
            )?;
        }

        // unwind the nested loops
        for i in (0..expr_rank).rev() {
            // update and store contract sum
            if i == expr_rank - contract_by - 1 && contract_sum.is_some() {
                let contract_strides = contract_strides.as_ref().unwrap();
                let elmt_index = indices
                    .iter()
                    .take(contract_strides.len())
                    .zip(contract_strides.iter())
                    .fold(zero, |acc, (i, s)| {
                        let tmp = self.builder.ins().imul(*i, *s);
                        self.builder.ins().iadd(acc, tmp)
                    });
                let contract_sum_value =
                    self.builder
                        .ins()
                        .stack_load(self.real_type, contract_sum.unwrap(), 0);

                self.jit_compile_store(name, elmt, elmt_index, contract_sum_value, translation)?;
            }

            // increment index
            let next_index = self.builder.ins().iadd(indices[i], one);
            let block = self.builder.create_block();
            let loop_cond = if i == 0 && self.threaded {
                self.builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, next_index, thread_end.unwrap())
            } else {
                self.builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, next_index, expr_shape[i])
            };

            self.builder
                .ins()
                .brif(loop_cond, blocks[i], &[next_index], block, &[]);
            self.builder.seal_block(blocks[i]);
            self.builder.seal_block(block);
            self.builder.switch_to_block(block);
        }
        if let Some(exit_block) = exit_block {
            self.builder.ins().jump(exit_block, &[]);
            self.builder.seal_block(exit_block);
            self.builder.switch_to_block(exit_block);
        }
        Ok(())
    }

    fn jit_compile_sparse_contraction_block(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        translation: &Translation,
        is_tangent: bool,
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

        let final_contract_index = self
            .builder
            .ins()
            .iconst(int_type, i64::try_from(elmt.layout().nnz()).unwrap());

        // we will thread the length of the contract index (the outer loop)
        let (thread_start, thread_end, exit_block) = if self.threaded {
            let (start, end, exit_block) = self.jit_threading_limits(final_contract_index);
            (Some(start), Some(end), Some(exit_block))
        } else {
            (None, None, None)
        };

        // initialise the contract sum
        let contract_sum_var = self.decl_stack_slot(self.real_type, None);

        // loop through each contraction
        let block = self.builder.create_block();
        let contract_index = self.builder.append_block_param(block, self.int_type);
        self.builder
            .ins()
            .jump(block, &[thread_start.unwrap_or(zero)]);
        self.builder.switch_to_block(block);

        // start and end indices stored next to each other in the indices array
        // start_index = translation_index + 2 * contract_index
        let translation_index_val = self
            .builder
            .ins()
            .iconst(int_type, i64::try_from(translation_index).unwrap());
        let double_contract_index = self.builder.ins().imul(two, contract_index);
        let start_index = self
            .builder
            .ins()
            .iadd(translation_index_val, double_contract_index);
        // end_index = start_index + 1
        let end_index = self.builder.ins().iadd(start_index, one);

        // index into the indices array to get the start and end indices
        // start_contract = indices[translation_index + 2 * contract_index]
        // end_contract = indices[translation_index + 2 * contract_index + 1]
        let indices_array = self
            .builder
            .ins()
            .global_value(self.int_ptr_type, self.indices);
        let ptr = self.ptr_add_offset(self.int_type, indices_array, start_index);
        let start_contract = self
            .builder
            .ins()
            .load(self.int_type, self.mem_flags, ptr, 0);
        let ptr = self.ptr_add_offset(self.int_type, indices_array, end_index);
        let end_contract = self
            .builder
            .ins()
            .load(self.int_type, self.mem_flags, ptr, 0);

        // init sum
        let fzero = self.fconst(0.0);
        self.builder.ins().stack_store(fzero, contract_sum_var, 0);

        // loop through each element in the contraction
        let contract_block = self.builder.create_block();
        let expr_index = self
            .builder
            .append_block_param(contract_block, self.int_type);
        self.builder.ins().jump(contract_block, &[start_contract]);
        self.builder.switch_to_block(contract_block);

        // loop body - load index from layout
        let rank_val = self.builder.ins().iconst(
            self.int_type,
            i64::try_from(elmt.expr_layout().rank()).unwrap(),
        );
        let elmt_index_mult_rank = self.builder.ins().imul(expr_index, rank_val);
        let indices_int = (0..elmt.expr_layout().rank())
            // index = indices[layout_index + i + elmt_index * rank]
            .map(|i| {
                let layout_index_plus_offset = self
                    .builder
                    .ins()
                    .iconst(self.int_type, i64::try_from(layout_index + i).unwrap());
                let curr_index = self
                    .builder
                    .ins()
                    .iadd(elmt_index_mult_rank, layout_index_plus_offset);
                let ptr = self.ptr_add_offset(self.int_type, indices_array, curr_index);
                let index = self
                    .builder
                    .ins()
                    .load(self.int_type, self.mem_flags, ptr, 0);
                Ok(index)
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        // loop body - eval expression and increment sum
        let expr = if is_tangent {
            elmt.tangent_expr()
        } else {
            elmt.expr()
        };
        let float_value =
            self.jit_compile_expr(name, expr, indices_int.as_slice(), elmt, Some(expr_index))?;
        let contract_sum_value = self
            .builder
            .ins()
            .stack_load(self.real_type, contract_sum_var, 0);
        let new_contract_sum_value = self.builder.ins().fadd(contract_sum_value, float_value);
        self.builder
            .ins()
            .stack_store(new_contract_sum_value, contract_sum_var, 0);

        // increment contract loop index
        let next_elmt_index = self.builder.ins().iadd(expr_index, one);

        // contract loop condition
        let loop_while =
            self.builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, next_elmt_index, end_contract);
        let post_contract_block = self.builder.create_block();
        self.builder.ins().brif(
            loop_while,
            contract_block,
            &[next_elmt_index],
            post_contract_block,
            &[],
        );
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
        let next_contract_index = self.builder.ins().iadd(contract_index, one);

        // outer loop condition
        let loop_while = self.builder.ins().icmp(
            IntCC::UnsignedLessThan,
            next_contract_index,
            thread_end.unwrap_or(final_contract_index),
        );
        let post_block = exit_block.unwrap_or(self.builder.create_block());
        self.builder
            .ins()
            .brif(loop_while, block, &[next_contract_index], post_block, &[]);
        self.builder.seal_block(block);
        self.builder.switch_to_block(post_block);
        self.builder.seal_block(post_block);

        Ok(())
    }

    // for sparse blocks we can loop through the non-zero elements and extract the index from the layout, then we compile the expression passing in this index
    fn jit_compile_sparse_block(
        &mut self,
        name: &str,
        elmt: &TensorBlock,
        translation: &Translation,
        is_tangent: bool,
    ) -> Result<()> {
        let int_type = self.int_type;

        let layout_index = self.layout.get_layout_index(elmt.expr_layout()).unwrap();

        // loop through the non-zero elements
        let zero = self.builder.ins().iconst(int_type, 0);
        let one = self.builder.ins().iconst(int_type, 1);
        let end_index = self
            .builder
            .ins()
            .iconst(int_type, i64::try_from(elmt.layout().nnz()).unwrap());

        // we will thread the length of the nnzs (the outer loop)
        let (thread_start, thread_end, exit_block) = if self.threaded {
            let (start, end, exit_block) = self.jit_threading_limits(end_index);
            (Some(start), Some(end), Some(exit_block))
        } else {
            (None, None, None)
        };

        let block = self.builder.create_block();
        let curr_index = self.builder.append_block_param(block, int_type);
        self.builder
            .ins()
            .jump(block, &[thread_start.unwrap_or(zero)]);
        self.builder.switch_to_block(block);

        // loop body - load index from layout
        let elmt_index = curr_index;
        let rank_val = self
            .builder
            .ins()
            .iconst(int_type, i64::try_from(elmt.expr_layout().rank()).unwrap());
        let elmt_index_mult_rank = self.builder.ins().imul(elmt_index, rank_val);
        let indices_int = (0..elmt.expr_layout().rank())
            // index = indices[layout_index + i + elmt_index * rank]
            .map(|i| {
                let layout_index_plus_offset = self
                    .builder
                    .ins()
                    .iconst(int_type, i64::try_from(layout_index + i).unwrap());
                let curr_index = self
                    .builder
                    .ins()
                    .iadd(elmt_index_mult_rank, layout_index_plus_offset);
                let indices_ptr = self
                    .builder
                    .ins()
                    .global_value(self.int_ptr_type, self.indices);
                let ptr = self.ptr_add_offset(self.int_type, indices_ptr, curr_index);
                let index = self
                    .builder
                    .ins()
                    .load(self.int_type, self.mem_flags, ptr, 0);
                Ok(index)
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        // loop body - eval expression
        let expr = if is_tangent {
            elmt.tangent_expr()
        } else {
            elmt.expr()
        };
        let float_value =
            self.jit_compile_expr(name, expr, indices_int.as_slice(), elmt, Some(elmt_index))?;

        self.jit_compile_broadcast_and_store(
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
        let loop_while = self.builder.ins().icmp(
            IntCC::UnsignedLessThan,
            next_index,
            thread_end.unwrap_or(end_index),
        );
        let post_block = exit_block.unwrap_or(self.builder.create_block());

        self.builder
            .ins()
            .brif(loop_while, block, &[next_index], post_block, &[]);
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
        is_tangent: bool,
    ) -> Result<()> {
        let int_type = self.int_type;

        // loop through the non-zero elements
        let zero = self.builder.ins().iconst(int_type, 0);
        let one = self.builder.ins().iconst(int_type, 1);
        let block = self.builder.create_block();
        let end_index = self
            .builder
            .ins()
            .iconst(int_type, i64::try_from(elmt.expr_layout().nnz()).unwrap());

        // we will thread the length of the nnzs (the outer loop)
        let (thread_start, thread_end, exit_block) = if self.threaded {
            let (start, end, exit_block) = self.jit_threading_limits(end_index);
            (Some(start), Some(end), Some(exit_block))
        } else {
            (None, None, None)
        };

        let curr_index = self.builder.append_block_param(block, int_type);
        self.builder
            .ins()
            .jump(block, &[thread_start.unwrap_or(zero)]);
        self.builder.switch_to_block(block);

        // loop body - index is just the same for each element
        let elmt_index = curr_index;
        let indices_int = vec![elmt_index; elmt.expr_layout().rank()];

        // loop body - eval expression
        let expr = if is_tangent {
            elmt.tangent_expr()
        } else {
            elmt.expr()
        };
        let float_value =
            self.jit_compile_expr(name, expr, indices_int.as_slice(), elmt, Some(elmt_index))?;

        // loop body - store result
        self.jit_compile_broadcast_and_store(
            name,
            elmt,
            float_value,
            elmt_index,
            translation,
            block,
        )?;

        // increment loop index
        let next_index = self.builder.ins().iadd(elmt_index, one);
        let loop_while = self.builder.ins().icmp(
            IntCC::UnsignedLessThan,
            next_index,
            thread_end.unwrap_or(end_index),
        );
        let post_block = exit_block.unwrap_or(self.builder.create_block());
        self.builder
            .ins()
            .brif(loop_while, block, &[next_index], post_block, &[]);
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
                let bcast_end_index = self
                    .builder
                    .ins()
                    .iconst(int_type, i64::try_from(broadcast_len).unwrap());
                let bcast_index = self.builder.append_block_param(bcast_block, self.int_type);

                // setup loop block
                self.builder.ins().jump(bcast_block, &[bcast_start_index]);
                self.builder.switch_to_block(bcast_block);

                // store value at index = expr_index * broadcast_len + bcast_index
                let tmp = self.builder.ins().imul(expr_index, bcast_end_index);
                let store_index = self.builder.ins().iadd(tmp, bcast_index);
                self.jit_compile_store(name, elmt, store_index, float_value, translation)?;

                // increment index
                let bcast_next_index = self.builder.ins().iadd(bcast_index, one);
                let bcast_cond = self.builder.ins().icmp(
                    IntCC::UnsignedLessThan,
                    bcast_next_index,
                    bcast_end_index,
                );
                let post_bcast_block = self.builder.create_block();
                self.builder.ins().brif(
                    bcast_cond,
                    bcast_block,
                    &[bcast_next_index],
                    post_bcast_block,
                    &[],
                );
                self.builder.seal_block(bcast_block);
                self.builder.seal_block(post_bcast_block);
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
        let res_index = match &translation.target {
            TranslationTo::Contiguous { start, end: _ } => {
                let start_const = self
                    .builder
                    .ins()
                    .iconst(int_type, i64::try_from(*start).unwrap());
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
                let translate_store_index = self
                    .builder
                    .ins()
                    .iconst(int_type, i64::try_from(translate_store_index).unwrap());
                let elmt_index_strided = store_index;
                let curr_index = self
                    .builder
                    .ins()
                    .iadd(elmt_index_strided, translate_store_index);
                let indices_ptr = self
                    .builder
                    .ins()
                    .global_value(self.int_ptr_type, self.indices);
                let ptr = self.ptr_add_offset(self.int_type, indices_ptr, curr_index);
                self.builder
                    .ins()
                    .load(self.int_type, self.mem_flags, ptr, 0)
            }
        };

        let ptr = self.ptr_add_offset(self.real_type, self.tensor_ptr.unwrap(), res_index);
        self.builder
            .ins()
            .store(self.mem_flags, float_value, ptr, 0);

        Ok(())
    }

    fn declare_variable(&mut self, ty: types::Type, name: &str, val: Value) -> Variable {
        let index = self.variables.len();
        let var = Variable::new(index);
        if !self.variables.contains_key(name) {
            self.variables.insert(name.into(), var);
            self.builder.declare_var(var, ty);
            self.builder.def_var(var, val);
        }
        var
    }

    fn get_tangent_tensor_name(&self, name: &str) -> String {
        format!("{name}__tangent__")
    }

    fn insert_tensor(&mut self, tensor: &Tensor, ptr: Value, data_index: i64, is_tangent: bool) {
        let mut tensor_data_index = data_index;
        let tensor_data_ptr = self.ptr_add_offset_i64(self.real_type, ptr, tensor_data_index);
        let tensor_name = if is_tangent {
            self.get_tangent_tensor_name(tensor.name())
        } else {
            tensor.name().to_owned()
        };
        self.declare_variable(self.real_ptr_type, tensor_name.as_str(), tensor_data_ptr);

        //insert any named blocks
        for blk in tensor.elmts() {
            if let Some(name) = blk.name() {
                let blk_name = if is_tangent {
                    self.get_tangent_tensor_name(name)
                } else {
                    name.to_owned()
                };
                let tensor_data_ptr =
                    self.ptr_add_offset_i64(self.real_type, ptr, tensor_data_index);
                self.declare_variable(self.real_ptr_type, blk_name.as_str(), tensor_data_ptr);
            }
            // named blocks only supported for rank <= 1, so we can just add the nnz to get the next data index
            tensor_data_index += i64::try_from(blk.nnz()).unwrap();
        }
    }

    pub fn new(
        module: &'ctx mut CraneliftModule<M>,
        model: &DiscreteModel,
        arg_names: &[&str],
        arg_types: &[Type],
    ) -> Self {
        module.ctx.func.signature.params.clear();
        module.ctx.func.signature.returns.clear();

        for ty in arg_types {
            module.ctx.func.signature.params.push(AbiParam::new(*ty));
        }

        // Create the builder to build a function.
        let mut builder = FunctionBuilder::new(&mut module.ctx.func, &mut module.builder_context);

        let indices = module
            .module
            .declare_data_in_func(module.indices_id, builder.func);

        let constants = module
            .module
            .declare_data_in_func(module.constants_id, builder.func);

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
            indices,
            constants,
            variables: HashMap::new(),
            mem_flags: MemFlags::new(),
            functions: HashMap::new(),
            layout: &module.layout,
            threaded: module.threaded,
        };

        // insert arg vars
        for (i, (arg_name, arg_type)) in arg_names.iter().zip(arg_types.iter()).enumerate() {
            let val = codegen.builder.block_params(entry_block)[i];
            codegen.declare_variable(*arg_type, arg_name, val);
        }

        // insert u if it exists in args
        if let Some(u) = codegen.variables.get("u") {
            let u_ptr = codegen.builder.use_var(*u);
            codegen.insert_tensor(model.state(), u_ptr, 0, false);
        }

        if let Some(du) = codegen.variables.get("du") {
            let du_ptr = codegen.builder.use_var(*du);
            codegen.insert_tensor(model.state(), du_ptr, 0, true);
        }

        // insert dudt if it exists in args and is used in the model
        if let Some(dudt) = codegen.variables.get("dudt") {
            if let Some(state_dot) = model.state_dot() {
                let statedot_ptr = codegen.builder.use_var(*dudt);
                codegen.insert_tensor(state_dot, statedot_ptr, 0, false);
            }
        }

        // insert out if it exists in args and is used in the model
        if let Some(out_var) = codegen.variables.get("out") {
            if let Some(out) = model.out() {
                let out_ptr = codegen.builder.use_var(*out_var);
                codegen.insert_tensor(out, out_ptr, 0, false);
            }
        }

        // insert dout if it exists in args and is
        if let Some(dout) = codegen.variables.get("dout") {
            if let Some(out) = model.out() {
                let dout_ptr = codegen.builder.use_var(*dout);
                codegen.insert_tensor(out, dout_ptr, 0, true);
            }
        }

        // todo: insert constant tensors

        let constants = codegen
            .builder
            .ins()
            .global_value(codegen.real_ptr_type, codegen.constants);
        for tensor in model.constant_defns() {
            let data_index =
                i64::try_from(codegen.layout.get_data_index(tensor.name()).unwrap()).unwrap();
            codegen.insert_tensor(tensor, constants, data_index, false);
        }

        // insert all tensors in data if it exists in args
        let tensors = model.inputs().iter();
        let tensors = tensors.chain(model.input_dep_defns().iter());
        let tensors = tensors.chain(model.time_dep_defns().iter());
        let tensors = tensors.chain(model.state_dep_defns().iter());

        if let Some(data) = codegen.variables.get("data") {
            let data_ptr = codegen.builder.use_var(*data);

            for tensor in tensors.clone() {
                let data_index =
                    i64::try_from(codegen.layout.get_data_index(tensor.name()).unwrap()).unwrap();
                codegen.insert_tensor(tensor, data_ptr, data_index, false);
            }
        }

        // insert all tangent tensors in tangent_data if it exists in args
        if let Some(data) = codegen.variables.get("ddata") {
            let data_ptr = codegen.builder.use_var(*data);

            for tensor in tensors {
                let data_index =
                    i64::try_from(codegen.layout.get_data_index(tensor.name()).unwrap()).unwrap();
                codegen.insert_tensor(tensor, data_ptr, data_index, true);
            }
        }
        codegen
    }

    fn jit_compile_inputs(
        &mut self,
        model: &DiscreteModel,
        base_data_ptr: Value,
        is_tangent: bool,
        is_get: bool,
    ) {
        let mut inputs_index = 0;
        for input in model.inputs() {
            let data_index =
                i64::try_from(self.layout.get_data_index(input.name()).unwrap()).unwrap();
            self.insert_tensor(input, base_data_ptr, data_index, is_tangent);
            let tensor_name = if is_tangent {
                self.get_tangent_tensor_name(input.name())
            } else {
                input.name().to_owned()
            };
            let data_ptr = self.variables.get(tensor_name.as_str()).unwrap();
            let data_ptr = self.builder.use_var(*data_ptr);
            let input_name = if is_tangent { "dinputs" } else { "inputs" };
            let input_ptr = self.variables.get(input_name).unwrap();
            let input_ptr = self.builder.use_var(*input_ptr);
            let inputs_start_index = self
                .builder
                .ins()
                .iconst(self.int_type, i64::try_from(inputs_index).unwrap());

            // loop thru the elements of this input and set them using the inputs ptr
            let start_index = self.builder.ins().iconst(self.int_type, 0);

            let input_block = self.builder.create_block();
            let curr_input_index = self.builder.append_block_param(input_block, self.int_type);
            self.builder.ins().jump(input_block, &[start_index]);
            self.builder.switch_to_block(input_block);

            // loop body - copy value from inputs to data
            let curr_input_index_plus_start_index = self
                .builder
                .ins()
                .iadd(curr_input_index, inputs_start_index);
            let indexed_input_ptr =
                self.ptr_add_offset(self.real_type, input_ptr, curr_input_index_plus_start_index);
            let indexed_data_ptr = self.ptr_add_offset(self.real_type, data_ptr, curr_input_index);
            if is_get {
                let input_value =
                    self.builder
                        .ins()
                        .load(self.real_type, self.mem_flags, indexed_data_ptr, 0);
                self.builder
                    .ins()
                    .store(self.mem_flags, input_value, indexed_input_ptr, 0);
            } else {
                let input_value =
                    self.builder
                        .ins()
                        .load(self.real_type, self.mem_flags, indexed_input_ptr, 0);
                self.builder
                    .ins()
                    .store(self.mem_flags, input_value, indexed_data_ptr, 0);
            }

            // increment loop index
            let one = self.builder.ins().iconst(self.int_type, 1);
            let next_index = self.builder.ins().iadd(curr_input_index, one);

            let loop_while = self.builder.ins().icmp_imm(
                IntCC::UnsignedLessThan,
                next_index,
                i64::try_from(input.nnz()).unwrap(),
            );
            let post_block = self.builder.create_block();
            self.builder
                .ins()
                .brif(loop_while, input_block, &[next_index], post_block, &[]);
            self.builder.seal_block(input_block);
            self.builder.seal_block(post_block);
            self.builder.switch_to_block(post_block);

            // get ready for next input
            inputs_index += input.nnz();
        }
    }
}
