use inkwell::intrinsics::Intrinsic;
use inkwell::types::{AnyType, BasicType, FloatType, BasicMetadataTypeEnum};
use inkwell::values::{PointerValue, FloatValue, FunctionValue, ArrayValue, IntValue, StructValue, VectorValue, FloatMathValue};
use inkwell::{OptimizationLevel, AddressSpace};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use sundials_sys::{realtype, N_Vector, IDACreate, N_VNew_Serial, N_VCloneVectorArray, N_VGetArrayPointer, N_VConst, IDAInit, IDASVtolerances, IDARootInit, IDASetUserData, IDASetJacFn, IDASetJacTimes, SUNLinSolInitialize, IDASetId, IDASensInit, IDASensEEtolerances, SUNMatrix, SUNLinearSolver, SUNSparseMatrix, SUNDenseMatrix, PREC_NONE, PREC_LEFT, SUNLinSol_Dense, SUNLinSol_SPBCGS, SUNLinSol_SPFGMR, SUNLinSol_SPGMR, SUNLinSol_SPTFQMR, IDASetLinearSolver, IDA_SIMULTANEOUS, IDASensFree, SUNLinSolFree, SUNMatDestroy, N_VDestroy, N_VDestroyVectorArray, IDAFree};
use std::cmp::max;
use std::collections::HashMap;
use std::error::{Error, self};
use std::ffi::c_void;
use std::fmt::Pointer;
use std::ptr::null;
use std::{io, fmt};
use std::iter::zip;

use crate::ast::{Ast, AstKind};
use crate::discretise::{DiscreteModel, Array};

type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

struct Args<'ctx> {
    time: FloatValue<'ctx>,
    y: PointerValue<'ctx>,
    yp: PointerValue<'ctx>,
    input: PointerValue<'ctx>,
    rr: PointerValue<'ctx>,
}

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type ResidualFunc = unsafe extern "C" fn(time: realtype, u: *const realtype, up: *const realtype, inputs: *const realtype, rr: *mut realtype);

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    model: &'ctx DiscreteModel<'ctx>,
    variables: HashMap<String, PointerValue<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    fn_value_opt: Option<FunctionValue<'ctx>>,
    real_type: FloatType<'ctx>,
    real_type_str: String,
}

impl<'ctx> CodeGen<'ctx> {
    fn get_function(&mut self, name: &str) -> Option<FunctionValue<'ctx>> {
        match self.function.get(name) {
            Some(func) => func,
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
                        for (i, arg) in fn_val.get_param_iter().enumerate() {
                            arg.into_float_value().set_name("x");
                        }
                        self.functions.insert(name.to_owned(), fn_val);
                    },
                    unknown => None,
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

    fn jit_compile_array(&mut self, a: &Array)  -> Result<PointerValue> {
        if a.dim == 1 {
            let float_value = self.jit_compile_expr(&a.elmts[0], a.name)?;
            let float_alloca = self.create_entry_block_builder().build_alloca(self.real_type, a.name);
            self.builder.build_store(float_alloca, float_value);
            Ok(float_alloca)
        } else {
            let array_type = self.real_type.array_type(a.dim);
            let array_alloca = self.create_entry_block_builder().build_alloca(array_type, a.name);
            let mut array = self.builder.build_load(array_alloca, a.name).into_array_value();
            for (i, elmt) in a.elmts.iter().enumerate() {
                let elmt_name = format!("{}-{}", a.name, i);
                if elmt.get_dim() == 1 {
                    let float_value = self.jit_compile_expr(&elmt.expr, elmt_name.as_str())?;
                    array = self.builder.build_insert_value(array, float_value, elmt.bounds.0, a.name).unwrap().into_array_value();
                } else {
                    let array_v = self.jit_compile_array_expr(&elmt.expr)?;
                    array = self.jit_compile_insert_slice(array, array_v, elmt.bounds, a.name)?;
                }
            }
            self.builder.build_store(array_alloca, array);
            Ok(array_alloca)
        }
    }

    fn jit_compile_expr(&mut self, expr: &Ast, name: &str) -> Result<FloatMathValue<'ctx>> {
        match expr.kind {
            AstKind::Binop(binop) => {
                let lhs = self.jit_compile_expr(binop.left.as_ref(), name);
                let rhs = self.jit_compile_expr(binop.right.as_ref(), name);
                match binop.op {
                    '*' => self.builder.build_float_mul(lhs, rhs, name),
                    '/' => self.builder.build_float_div(lhs, rhs, name),
                    '-' => self.builder.build_float_sub(lhs, rhs, name),
                    '+' => self.builder.build_float_add(lhs, rhs, name),
                    _ => Err("unknown binop op")
                }
            },
            AstKind::Monop(monop) => {
                let child = self.jit_compile_expr(monop.child.as_ref(), name);
                match monop.op {
                    '-' => self.builder.build_float_neg(child, name),
                    _ => Err("unknown monop op")
                }                
            },
            AstKind::Call(call) => {
                match self.get_function(call.fn_name) {
                    Some(function) => {
                        let mut args: Vec<FloatMathValue> = Vec::new();
                        for i in 0..function.count_params() {
                            args.push(self.jit_compile_expr(call.args[i]?, name));
                        }
                        self.builder.build_call(function, args, name)
                    },
                    _ => Err("unknown function call")
                }
                
            },
            AstKind::CallArg(arg) => {
                self.jit_compile_expr(&arg.expression, name)
            },
            AstKind::Index(index) => index.,
            AstKind::Slice(_) => todo!(),
            AstKind::Number(_) => todo!(),
            AstKind::Integer(_) => todo!(),
            AstKind::Name(_) => todo!(),
            _ => panic!("unexprected astkind"),
        }
    }
    
    fn jit_compile_array_expr(&mut self, expr: &Ast) -> Result<ArrayValue<'ctx>> {
        todo!()
    }
    fn jit_compile_insert_slice(&mut self, to_array: ArrayValue<'ctx>, from_array: ArrayValue<'ctx>, bounds: (u32, u32), name: &str) -> Result<ArrayValue<'ctx>> {
        todo!()
    }

    fn jit_compile_residual(&mut self) -> Result<JitFunction<ResidualFunc>> {
        let real_ptr_type = self.real_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(
            &[self.real_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into(), real_ptr_type.into()]
            , false
        );
        let n_states = self.model.len_state();
        let n_inputs = self.model.len_inputs();
        let fn_arg_names = &["t", "u", "dotu", "inputs", "rr"];
        let fn_arg_dims = &[1, n_states, n_states, n_inputs, n_states];
        let function = self.module.add_function("residual", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        for (i, arg) in function.get_param_iter().enumerate() {
            let arg_name = fn_arg_names[i];
            let arg_dim = fn_arg_dims[i];
            let alloca = match arg {
                inkwell::values::BasicValueEnum::PointerValue(v) => v,
                inkwell::values::BasicValueEnum::FloatValue(v) => {
                    let alloca = self.create_entry_block_builder().build_alloca(arg.get_type(), arg_name);
                    self.builder.build_store(alloca, arg);
                    alloca
                }
                _ => unreachable!()
            };
            self.variables.insert(arg_name.to_owned(), Local {ptr: alloca, dim: arg_dim});
        }

        self.builder.position_at_end(basic_block);

        // input definitiions
        for a in self.model.in_defns {
            let alloca = self.jit_compile_array(a)?;
            self.variables.insert(a.name, alloca);
        }
        // F and G
        let lhs = self.jit_compile_array(self.model.lhs)?;
        let rhs = self.jit_compile_array(self.model.rhs)?;
        
        // compute residual here
        todo!()

        self.builder.build_return(None);

        unsafe { self.execution_engine.get_function("residual").ok() }
    }
    fn residual(&self, args: &Args) {

    }
}

struct Options {
    atol: f64,
    rtol: f64,
    print_stats: bool,
    using_sparse_matrix: bool,
    using_iterative_solver: bool,
    jacobian: String,
    linear_solver: String, // klu, lapack, spbcg 
    preconditioner: String, // spbcg 
    linsol_max_iterations: i32,
    precon_half_bandwidth: i32,
    precon_half_bandwidth_keep: i32,
}

struct Sundials<'ctx> {
    ida_mem: *const c_void, // pointer to memory
    number_of_states: i32,
    number_of_parameters: i32,
    yy: N_Vector,
    yp: N_Vector, 
    avtol: N_Vector,
    yyS: Vec<N_Vector>, 
    ypS: Vec<N_Vector>,
    id: N_Vector,
    rtol: realtype,
    jacobian: SUNMatrix,
    linear_solver: SUNLinearSolver, 
    residual: JitFunction<'ctx, ResidualFunc>,
    options: Options,
}

impl<'ctx> Sundials<'ctx> {
    pub fn from_discrete_model(model: DiscreteModel, options: Options) -> Result<Sundials> {
        let number_of_states = i64::try_from(model.len_state())?;
        let number_of_parameters = i64::try_from(model.len_inputs())?;
        let context = Context::create();
        let module = context.create_module(model.name);
        let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;
        let real_type = context.f64_type();
        let real_type_str = "f64";
        let codegen = CodeGen {
            context: &context,
            module,
            builder: context.create_builder(),
            execution_engine,
            model,
            real_type,
            real_type_str,
        };
        let residual = codegen.jit_compile_residual()?;

        unsafe {
            let ida_mem = IDACreate();

            // allocate vectors
            let yy = N_VNew_Serial(number_of_states);
            let yp = N_VNew_Serial(number_of_states);
            let avtol = N_VNew_Serial(number_of_states);
            let id = N_VNew_Serial(number_of_states);

            let yyS: Vec<N_Vector> = Vec::new();
            let ypS: Vec<N_Vector> = Vec::new();
            for is in 0..number_of_parameters {
                yyS.push(N_VNew_Serial(number_of_parameters));
                ypS.push(N_VNew_Serial(number_of_parameters));
            }

            // set tolerances
            let rtol = options.rtol;
            N_VConst(options.atol, avtol);

            for (yySi, ypSi) in zip(yyS, ypS) {
                N_VConst(0.0, yySi);
                N_VConst(0.0, ypSi);
            }

            // initialise solver
            IDAInit(ida_mem, residual, 0, yy, yp);

            // set tolerances
            IDASVtolerances(ida_mem, options.rtol, avtol);

            // set events
            //IDARootInit(ida_mem, number_of_events, events_casadi);


            // set matrix
            let jacobian = if options.jacobian == "sparse" {
                return Err("sparse jacobian not implemented".into())
            }
            else if options.jacobian == "dense" || options.jacobian == "none" {
                SUNDenseMatrix(number_of_states, number_of_states)
            }
            else if options.jacobian == "matrix-free" {
                null()
            } else {
                return Err(format!("unknown jacobian {}", options.jacobian).into())
            };

            let precon_type = if options.preconditioner == "none" {
                PREC_NONE
            } else {
                PREC_LEFT
            };

            // set linear solver
            let linear_solver = if options.linear_solver == "SUNLinSol_Dense" {
                SUNLinSol_Dense(yy, jacobian)
            }
            else if options.linear_solver == "SUNLinSol_KLU" {
                return Err("KLU linear solver not implemented".into())
            }
            else if options.linear_solver == "SUNLinSol_SPBCGS" {
                SUNLinSol_SPBCGS(yy, precon_type, options.linsol_max_iterations)
            }
            else if options.linear_solver == "SUNLinSol_SPFGMR" {
                SUNLinSol_SPFGMR(yy, precon_type, options.linsol_max_iterations)
            }
            else if options.linear_solver == "SUNLinSol_SPGMR" {
                SUNLinSol_SPGMR(yy, precon_type, options.linsol_max_iterations)
            }
            else if options.linear_solver == "SUNLinSol_SPTFQMR" {
                SUNLinSol_SPTFQMR(yy, precon_type, options.linsol_max_iterations)
            } else {
                return Err(format!("unknown linear solver {}", options.linear_solver).into())
            };

            IDASetLinearSolver(ida_mem, linear_solver, jacobian);

            if options.preconditioner != "none" {
                return Err("preconditioner not implemented")
            }

            if options.jacobian == "matrix-free" {
                IDASetJacTimes(ida_mem, null, jtimes);
            }
            else if options.jacobian != "none" {
                IDASetJacFn(ida_mem, jacobian_casadi);
            }

            if number_of_parameters > 0 {
                IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
                            sensitivities, yyS, ypS);
                IDASensEEtolerances(ida_mem);
            }

            SUNLinSolInitialize(linear_solver);

            let id_val = N_VGetArrayPointer(id);
            for (ii, state) in model.states.iter().enumerate() {
                id_val[ii] = if state.is_algebraic() { 0 } else { 1 };
            }

            IDASetId(ida_mem, id);
            
            let sundials = Sundials {
                ida_mem,
                number_of_states,
                number_of_parameters,
                yy,
                yp,
                avtol,
                yyS,
                ypS,
                id,
                rtol,
                jacobian,
                linear_solver,
                residual,
                options,
            };
            IDASetUserData(sundials.ida_mem, &sundials);
            sundials
        }
    }

    pub fn solve(&self) {
    }

    pub fn destroy(&mut self) {
        unsafe {
             /* Free memory */
            if self.number_of_parameters > 0 {
                IDASensFree(self.ida_mem);
            }
            SUNLinSolFree(self.linear_solver);
            SUNMatDestroy(self.jacobian);
            N_VDestroy(self.avtol);
            N_VDestroy(self.yy);
            N_VDestroy(self.yp);
            N_VDestroy(self.id);
            for (yySi, ypSi) in zip(self.yyS, self.ypS) {
                N_VDestroy(yySi);
                N_VDestroy(ypSi);
            }
            IDAFree(&self.ida_mem);
        }
    }
}