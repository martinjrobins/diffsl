use inkwell::intrinsics::Intrinsic;
use inkwell::passes::PassManager;
use inkwell::types::{FloatType, BasicMetadataTypeEnum, BasicTypeEnum};
use inkwell::values::{PointerValue, FloatValue, FunctionValue, IntValue, BasicMetadataValueEnum, BasicValueEnum};
use inkwell::{OptimizationLevel, AddressSpace, IntPredicate};
use inkwell::builder::Builder;
use inkwell::execution_engine::{ExecutionEngine, JitFunction, UnsafeFunctionPointer};
use inkwell::module::Module;
use ndarray::{Array1, Array2, Array3, ShapeBuilder};
use sundials_sys::{realtype, N_Vector, IDAGetSens, IDAGetNonlinSolvStats, IDA_TSTOP_RETURN, IDA_SUCCESS, IDA_ROOT_RETURN, IDA_YA_YDP_INIT, IDA_NORMAL, IDASolve, IDAGetIntegratorStats, IDASetStopTime, IDACreate, N_VNew_Serial, N_VGetArrayPointer, N_VConst, IDAInit, IDACalcIC, IDASVtolerances, IDASetUserData, SUNLinSolInitialize, IDASetId, SUNMatrix, SUNLinearSolver, SUNDenseMatrix, PREC_NONE, PREC_LEFT, SUNLinSol_Dense, SUNLinSol_SPBCGS, SUNLinSol_SPFGMR, SUNLinSol_SPGMR, SUNLinSol_SPTFQMR, IDASetLinearSolver, IDASensFree, SUNLinSolFree, SUNMatDestroy, N_VDestroy, IDAFree, IDAReInit};
use std::collections::HashMap;
use std::vec;
use std::ffi::c_void;
use std::ptr::{null_mut};
use std::iter::{zip};
use anyhow::{Result, anyhow};


use crate::ast::{Ast, AstKind};
use crate::discretise::{DiscreteModel, Array, ArrayElmt, Input};

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type ResidualFunc = unsafe extern "C" fn(time: realtype, u: *const realtype, up: *const realtype, inputs: *const realtype, rr: *mut realtype);
type U0Func = unsafe extern "C" fn(inputs: *const realtype, u: *mut realtype, up: *mut realtype);

struct CodeGen<'ctx> {
    context: &'ctx inkwell::context::Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    fpm: PassManager<FunctionValue<'ctx>>,
    ee: ExecutionEngine<'ctx>,
    variables: HashMap<String, PointerValue<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    fn_value_opt: Option<FunctionValue<'ctx>>,
    real_type: FloatType<'ctx>,
    real_type_str: String,
}

impl<'ctx> CodeGen<'ctx> {
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
    
    
    fn jit_compile_scalar_array(&mut self, a: &Array, res_ptr_opt: Option<PointerValue<'ctx>>)  -> Result<PointerValue<'ctx>> {
        let res_type = self.real_type;
        let res_ptr = match res_ptr_opt {
            Some(ptr) => ptr,
            None => self.create_entry_block_builder().build_alloca(res_type, a.name),
        };
        let elmt = a.elmts.first().unwrap();
        let float_value = self.jit_compile_expr(&elmt.expr, None, a.name)?;
        self.builder.build_store(res_ptr, float_value);
        Ok(res_ptr)
    }

    fn jit_compile_array(&mut self, a: &Array, res_ptr_opt: Option<PointerValue<'ctx>>)  -> Result<PointerValue<'ctx>> {
        let a_dim = a.get_dim();
        if a_dim == 1 {
            return self.jit_compile_scalar_array(a, res_ptr_opt)
        }
        let a_dim_val = self.context.i32_type().const_int(u64::from(a_dim), false);
        let res_ptr = match res_ptr_opt {
            Some(ptr) => ptr,
            None => self.create_entry_block_builder().build_array_alloca(self.real_type, a_dim_val, a.name),
        };
        let one = self.context.i32_type().const_int(1, false);
        let mut res_index = self.context.i32_type().const_int(0, false);
        for (i, elmt) in a.elmts.iter().enumerate() {
            let elmt_name_string = format!("{}-{}", a.name, i);
            let elmt_name= elmt_name_string.as_str();
            let elmt_dim = elmt.get_dim();
            if elmt_dim < 2 {
                for _ in 0..elmt_dim {
                    let float_value = self.jit_compile_expr(&elmt.expr, Some(res_index), elmt_name)?;
                    let resi_ptr = unsafe { self.builder.build_in_bounds_gep(res_ptr, &[res_index], elmt_name) };
                    self.builder.build_store(resi_ptr, float_value);
                    res_index = self.builder.build_int_add(res_index, one, elmt_name);
                }
            } else {
                let block = self.context.append_basic_block(self.fn_value(), elmt_name);
                self.builder.build_unconditional_branch(block);
                self.builder.position_at_end(block);
                let final_index = self.context.i32_type().const_int(elmt.bounds.1.into(), false);
                let float_value = self.jit_compile_expr(&elmt.expr, Some(res_index), elmt_name)?;
                let resi_ptr = unsafe { self.builder.build_in_bounds_gep(res_ptr, &[res_index], elmt_name) };
                res_index = self.builder.build_int_add(res_index, one, elmt_name);
                self.builder.build_store(resi_ptr, float_value);
                let loop_while = self.builder.build_int_compare(IntPredicate::ULE, res_index, final_index, elmt_name);
                let after_block = self.context.append_basic_block(self.fn_value(), elmt_name);
                self.builder.build_conditional_branch(loop_while, block, after_block);
                self.builder.position_at_end(after_block);
            }
        }
        Ok(res_ptr)
    }

    fn jit_compile_expr(&mut self, expr: &Ast, index: Option<IntValue<'ctx>>, name: &str) -> Result<FloatValue<'ctx>> {
        match &expr.kind {
            AstKind::Binop(binop) => {
                let lhs = self.jit_compile_expr(binop.left.as_ref(), index, name)?;
                let rhs = self.jit_compile_expr(binop.right.as_ref(), index, name)?;
                match binop.op {
                    '*' => Ok(self.builder.build_float_mul(lhs, rhs, name)),
                    '/' => Ok(self.builder.build_float_div(lhs, rhs, name)),
                    '-' => Ok(self.builder.build_float_sub(lhs, rhs, name)),
                    '+' => Ok(self.builder.build_float_add(lhs, rhs, name)),
                    unknown => Err(anyhow!("unknown binop op '{}'", unknown))
                }
            },
            AstKind::Monop(monop) => {
                let child = self.jit_compile_expr(monop.child.as_ref(), index, name)?;
                match monop.op {
                    '-' => Ok(self.builder.build_float_neg(child, name)),
                    unknown => Err(anyhow!("unknown monop op '{}'", unknown))
                }                
            },
            AstKind::Call(call) => {
                // deal with dot(name)
                if call.fn_name == "dot" && call.args.len() == 1 {
                    if let AstKind::Name(name) = call.args[0].kind {
                        let ptr = self.variables.get(format!("dot({})", name).as_str()).expect("variable not found");
                        let ptr = match index {
                            Some(i) => unsafe { self.builder.build_in_bounds_gep(*ptr, &[i], name) },
                            None => *ptr,
                        };
                        return Ok(self.builder.build_load(ptr, name).into_float_value())
                    }
                }
                match self.get_function(call.fn_name) {
                    Some(function) => {
                        let mut args: Vec<BasicMetadataValueEnum> = Vec::new();
                        for arg in call.args.iter() {
                            let arg_val = self.jit_compile_expr(arg.as_ref(), index, name)?;
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
                self.jit_compile_expr(&arg.expression, index, name)
            },
            AstKind::Number(value) => Ok(self.real_type.const_float(*value)),
            AstKind::Name(name) => {
                let ptr = self.variables.get(*name).expect("variable not found");
                let ptr = match index {
                    Some(i) => unsafe { self.builder.build_in_bounds_gep(*ptr, &[i], name) },
                    None => *ptr,
                };
                Ok(self.builder.build_load(ptr, name).into_float_value())
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
    
    fn input_alloca(&mut self, input: &Input) -> PointerValue<'ctx> {
        let ptr = self.variables.get("inputs").unwrap();
        let i = self.context.i32_type().const_int(u64::from(input.bounds.0), false);
        unsafe { self.builder.build_in_bounds_gep(*ptr, &[i], input.name) }
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
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &[ "inputs", "u0", "dotu0"];
        let function = self.module.add_function("set_u0", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);
        self.builder.position_at_end(basic_block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.variables.insert(name.to_owned(), alloca);
        }

        for input in model.inputs.iter() {
            let alloca = self.input_alloca(input);
            self.variables.insert(input.name.to_owned(), alloca);
        }

        for a in model.in_defns.iter() {
            let alloca = self.jit_compile_array(a, None)?;
            self.variables.insert(a.name.to_owned(), alloca);
        }

        let u0_ptr = self.variables.get("u0").unwrap();
        let u0_array = model.get_init_state();
        self.jit_compile_array(&u0_array, Some(*u0_ptr))?;
        self.builder.build_return(None);

        if function.verify(true) {
            self.fpm.run_on(&function);

            Ok(function)
        } else {
            unsafe {
                function.delete();
            }
            Err(anyhow!("Invalid generated function."))
        }
    }

    fn compile_residual<'m>(& mut self, model: &'m DiscreteModel) -> Result<FunctionValue<'ctx>> {
        self.clear();
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let n_states = model.len_state();
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[self.real_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let fn_arg_names = &["t", "u", "dotu", "inputs", "rr"];
        let function = self.module.add_function("residual", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.fn_value_opt = Some(function);
        self.builder.position_at_end(basic_block);

        for (i, arg) in function.get_param_iter().enumerate() {
            let name = fn_arg_names[i];
            let alloca = self.function_arg_alloca(name, arg);
            self.variables.insert(name.to_owned(), alloca);
        }

        for input in model.inputs.iter() {
            let alloca = self.input_alloca(input);
            self.variables.insert(input.name.to_owned(), alloca);
        }

        // state variables
        for s in model.states.iter() {
            let ptr = self.variables.get("u").unwrap();
            let i = self.context.i32_type().const_int(u64::from(s.bounds.0), false);
            let alloca = unsafe { self.builder.build_in_bounds_gep(*ptr, &[i], s.name) };

            let ptr = self.variables.get("dotu").unwrap();
            let i = self.context.i32_type().const_int(u64::from(s.bounds.0), false);
            let alloca_dot = unsafe { self.builder.build_in_bounds_gep(*ptr, &[i], s.name) };

            self.variables.insert(s.name.to_owned(), alloca);
            self.variables.insert(format!("dot({})", s.name), alloca_dot);
        }

        // input definitiions
        for a in model.in_defns.iter() {
            let alloca = self.jit_compile_array(a, None)?;
            self.variables.insert(a.name.to_owned(), alloca);
        }
        // F and G
        let lhs_ptr = self.jit_compile_array(&model.lhs, None)?;
        self.variables.insert("F".to_owned(), lhs_ptr);
        let rhs_ptr = self.jit_compile_array(&model.rhs, None)?;
        self.variables.insert("G".to_owned(), rhs_ptr);
        
        // compute residual here as dummy array
        let residual = Array {
            name: "residual",
            elmts: vec![
                ArrayElmt { 
                    bounds: (0, n_states), 
                    expr: Ast { kind: AstKind::new_binop(
                                        '-', 
                                        Ast { kind: AstKind::new_name("F"), span: None }, 
                                        Ast { kind: AstKind::new_name("G"), span: None }
                                ),
                                span: None
                            }
                }
            ],
        };
        let res_ptr = self.variables.get("rr").unwrap();
        let _res_ptr = self.jit_compile_array(&residual, Some(*res_ptr))?;
        self.builder.build_return(None);

        if function.verify(true) {
            self.fpm.run_on(&function);

            Ok(function)
        } else {
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
    yy: N_Vector,
    yp: N_Vector, 
    avtol: N_Vector,
    inputs: N_Vector,
    yy_s: Vec<N_Vector>, 
    yp_s: Vec<N_Vector>,
    id: N_Vector,
    jacobian: SUNMatrix,
    linear_solver: SUNLinearSolver, 
    residual: JitFunction<'ctx, ResidualFunc>,
    set_u0: JitFunction<'ctx, U0Func>,
    options: Options,
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
            N_VGetArrayPointer(data.inputs), 
            N_VGetArrayPointer(rr), 
        );
        0
    }   

    pub fn calc_u0(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let number_of_inputs = inputs.len();
        assert_eq!(number_of_inputs, self.data.number_of_parameters);
        let mut u0 = Array1::zeros(self.data.number_of_states);
        unsafe {
            let inputs_ptr = N_VGetArrayPointer(self.data.inputs);
            for i in 0..number_of_inputs {
                *inputs_ptr.add(i) = inputs[i]; 
            }
            self.data.set_u0.call(N_VGetArrayPointer(self.data.inputs), N_VGetArrayPointer(self.data.yy), N_VGetArrayPointer(self.data.yp));
            let yy_ptr = N_VGetArrayPointer(self.data.yy);
            for i in 0..self.data.number_of_states {
                u0[i] = *yy_ptr.add(i); 
            }
        }
        u0
    }

    pub fn calc_residual(&self, t: f64, inputs: &Array1<f64>, u0: &Array1<f64>, up0: &Array1<f64>) -> Array1<f64> {
        let number_of_inputs = inputs.len();
        let number_of_states = u0.len();
        assert_eq!(number_of_states, up0.len());
        assert_eq!(number_of_inputs, self.data.number_of_parameters);
        assert_eq!(number_of_states, self.data.number_of_states);
        let mut res = Array1::zeros(number_of_states);
        unsafe {
            let rr = N_VNew_Serial(i64::try_from(number_of_states).unwrap());
            let inputs_ptr = N_VGetArrayPointer(self.data.inputs);
            for i in 0..number_of_inputs {
                *inputs_ptr.add(i) = inputs[i]; 
            }
            let u0_ptr = N_VGetArrayPointer(self.data.yy);
            let up0_ptr = N_VGetArrayPointer(self.data.yp);
            for i in 0..number_of_states {
                *u0_ptr.add(i) = u0[i]; 
                *up0_ptr.add(i) = up0[i]; 
            }
            self.data.residual.call(t, 
                N_VGetArrayPointer(self.data.yy), 
                N_VGetArrayPointer(self.data.yp), 
                N_VGetArrayPointer(self.data.inputs), 
                N_VGetArrayPointer(rr), 
            );
            let rr_ptr = N_VGetArrayPointer(rr);
            for i in 0..self.data.number_of_states {
                res[i] = *rr_ptr.add(i); 
            }
        }
        res
    }

    pub fn from_discrete_model<'m>(model: &'m DiscreteModel, context: &'ctx inkwell::context::Context, options: Options) -> Result<Sundials<'ctx>> {
        let number_of_states = i64::try_from(model.len_state()).unwrap();
        let number_of_parameters = i64::try_from(model.len_inputs()).unwrap();
        let module = context.create_module(model.name);
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
        };

        let set_u0 = codegen.compile_set_u0(model)?;
        let residual = codegen.compile_residual(model)?;
        print!("set_u0 ir is:");
        set_u0.print_to_stderr();
        print!("residual ir is:");
        residual.print_to_stderr();

        let set_u0 = codegen.jit::<U0Func>(set_u0)?;
        let residual = codegen.jit::<ResidualFunc>(residual)?;
        
        unsafe {
            let ida_mem = IDACreate();

            // allocate vectors
            let yy = N_VNew_Serial(number_of_states);
            let yp = N_VNew_Serial(i64::from(number_of_states));
            let avtol = N_VNew_Serial(i64::from(number_of_states));
            let id = N_VNew_Serial(i64::from(number_of_states));
            let inputs = N_VNew_Serial(i64::from(number_of_parameters));

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
            IDAInit(ida_mem, Some(Self::sresidual), 0.0, yy, yp);

            // set tolerances
            IDASVtolerances(ida_mem, options.rtol, avtol);

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

            IDASetLinearSolver(ida_mem, linear_solver, jacobian);

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
            for (ii, state) in model.states.iter().enumerate() {
                *id_val.add(ii) = if state.is_algebraic() { 0.0 } else { 1.0 };
            }

            IDASetId(ida_mem, id);

            let mut data = Box::new(
                SundialsData {
                    number_of_states: usize::try_from(number_of_states).unwrap(),
                    number_of_parameters: usize::try_from(number_of_parameters).unwrap(),
                    yy,
                    yp,
                    avtol,
                    inputs,
                    yy_s,
                    yp_s,
                    id,
                    jacobian,
                    linear_solver,
                    residual,
                    set_u0,
                    options,
                }
            );
            IDASetUserData(ida_mem, &mut *data as *mut _ as *mut c_void);
            let sundials = Sundials {
                ida_mem,
                data,
            };
            Ok(sundials)
        }
    }

    pub fn solve(&mut self, times: &Array1<f64>, inputs: &Array1<f64>) -> (Array1<f64>, Array2<f64>, Array3<f64>) {
        let number_of_timesteps = times.len();
        let number_of_inputs = inputs.len();
        assert_eq!(number_of_inputs, self.data.number_of_parameters);

        let mut t_return = Array1::zeros((number_of_timesteps).f());
        let mut y_return = Array2::zeros((number_of_timesteps, self.data.number_of_states).f());
        let mut y_s_return = Array3::zeros((number_of_timesteps, self.data.number_of_parameters, self.data.number_of_states).f());

        unsafe {
            let inputs_ptr = N_VGetArrayPointer(self.data.inputs);
            for (i, &v) in inputs.iter().enumerate() {
                *inputs_ptr.add(i) = v; 
            }

            let yval = N_VGetArrayPointer(self.data.yy);
            let ypval = N_VGetArrayPointer(self.data.yp);
            let inputsval = N_VGetArrayPointer(self.data.inputs);
            let mut ys_val: Vec<*mut f64> = Vec::new();
            for is in 0..self.data.number_of_parameters {
                ys_val.push(N_VGetArrayPointer(self.data.yy_s[is]));
                N_VConst(0.0, self.data.yy_s[is]);
                N_VConst(0.0, self.data.yp_s[is]);
            }

            self.data.set_u0.call(inputsval, yval, ypval);

            let t0 = times[0];

            IDAReInit(self.ida_mem, t0, self.data.yy, self.data.yp);



            t_return[0] = times[0];
            for j in 0..self.data.number_of_states {
                y_return[[0, j]] = *yval.add(j);
            }
            for j in 0..self.data.number_of_parameters {
                for k in 0..self.data.number_of_states {
                    y_s_return[[0, j, k]] = *ys_val[j].add(k);
                }
            }

            IDACalcIC(self.ida_mem, IDA_YA_YDP_INIT, times[1]);

            let mut retval: i32;
            let t_final = times.last().unwrap().clone();
            for t_i in 1..number_of_timesteps {
                let t_next = times[t_i];
                IDASetStopTime(self.ida_mem, t_next);
                let mut tret: realtype = 0.0;
                retval = IDASolve(self.ida_mem, t_final, & mut tret as *mut realtype, self.data.yy, self.data.yp, IDA_NORMAL);

                if retval == IDA_TSTOP_RETURN || retval == IDA_SUCCESS ||
                    retval == IDA_ROOT_RETURN {
                    //if self.data.number_of_parameters > 0 {
                    //    IDAGetSens(self.ida_mem, & mut tret as *mut realtype, self.data.yy_s.as_mut_ptr());
                    //}

                    t_return[t_i] = tret;
                    for j in 0..self.data.number_of_states {
                        y_return[[t_i, j]] = *yval.add(j);
                    }
                    //for j in 0..self.data.number_of_parameters {
                    //    for k in 0..self.data.number_of_states {
                    //        y_s_return[[t_i, j, k]] = *ys_val[j].add(k);
                    //    }
                    //}
                    if retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN {
                        break;
                    }
                } else {
                    // failed
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

                IDAGetIntegratorStats(self.ida_mem, 
                                    &mut nsteps as *mut i64, 
                                    &mut nrevals as *mut i64, 
                                    &mut nlinsetups as *mut i64, 
                                    &mut netfails as *mut i64,
                                    &mut klast as *mut i32, 
                                    &mut kcur as *mut i32, 
                                    &mut hinused as *mut f64, 
                                    &mut hlast as *mut f64,
                                    &mut hcur as *mut f64, 
                                    &mut tcur as *mut f64);

                    
                let mut nniters = 0_i64;
                let mut nncfails = 0_i64;
                IDAGetNonlinSolvStats(self.ida_mem, &mut nniters as *mut i64, &mut nncfails as *mut i64);

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

        return (t_return, y_return, y_s_return);
    }

    pub fn destroy(&mut self) {
        unsafe {
             /* Free memory */
            if self.data.number_of_parameters > 0 {
                IDASensFree(self.ida_mem);
            }
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
use approx::{assert_relative_eq};
use ndarray::{Array, array, s};

use crate::{parser::parse_string, discretise::DiscreteModel, builder::ModelInfo, codegen::{Sundials, Options}};
     #[test]
    fn rate_equationn() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t) ) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.output.len(), 0);
        let discrete = DiscreteModel::from(model_info);
        println!("{}", discrete);
        let options = Options::new();
        let context = inkwell::context::Context::create();
        let mut sundials = Sundials::from_discrete_model(&discrete, &context, options).unwrap();

        let times = Array::linspace(0., 1., 5);
        let r = 1.0;
        let k = 1.0;
        let y0 = 1.0;
        let inputs = array![r, k];

        // test set_u0
        let u0 = sundials.calc_u0(&inputs);
        let check_u0 = array![y0];
        assert_relative_eq!(u0, check_u0);

        // test residual
        let up0 = array![r * y0 * (1. - y0 / k)];
        let res = sundials.calc_residual(0., &inputs, &u0, &up0);
        let res_check = array![0.];
        assert_relative_eq!(res, res_check);

        // solve
        let (out_times, y, _yp) = sundials.solve(&times, &inputs);

        assert_relative_eq!(out_times, times);
        let y_check = k / ((k - y0) * (-r * times).mapv(f64::exp) / y0 + 1.);
        assert_relative_eq!(y_check, y.slice(s![.., 0]));

        sundials.destroy();
    }
}
 