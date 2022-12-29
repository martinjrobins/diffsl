use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use sundials_sys::{realtype, N_Vector, IDACreate, N_VNew_Serial, N_VCloneVectorArray, N_VGetArrayPointer, N_VConst, IDAInit, IDASVtolerances, IDARootInit, IDASetUserData, IDASetJacFn, IDASetJacTimes, SUNLinSolInitialize, IDASetId, IDASensInit, IDASensEEtolerances, SUNMatrix, SUNLinearSolver, SUNSparseMatrix, SUNDenseMatrix, PREC_NONE, PREC_LEFT, SUNLinSol_Dense, SUNLinSol_SPBCGS, SUNLinSol_SPFGMR, SUNLinSol_SPGMR, SUNLinSol_SPTFQMR, IDASetLinearSolver, IDA_SIMULTANEOUS, IDASensFree, SUNLinSolFree, SUNMatDestroy, N_VDestroy, N_VDestroyVectorArray, IDAFree};
use std::error::Error;
use std::ffi::c_void;
use std::ptr::null;
use std::{io, fmt};
use std::iter::zip;

use crate::discretise::DiscreteModel;


type Result<T> = std::result::Result<T, DiffEqError>;

#[derive(Debug, Clone)]
struct DiffEqError {
    msg: &'static str
}

impl fmt::Display for DiffEqError{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}


#[repr(C)]
struct UserData {
    n_states: i64,
    inputs: *const realtype,
}

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type ResidualFunc = unsafe extern "C" fn(time: realtype, y: *const realtype, yp: *const realtype, rr: *mut realtype, data: *const UserData);

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_residual(&self) -> Result<JitFunction<ResidualFunc>> {
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
        let function = self.module.add_function("sum", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let x = function.get_nth_param(0)?.into_int_value();
        let y = function.get_nth_param(1)?.into_int_value();
        let z = function.get_nth_param(2)?.into_int_value();

        let sum = self.builder.build_int_add(x, y, "sum");
        let sum = self.builder.build_int_add(sum, z, "sum");

        self.builder.build_return(Some(&sum));

        unsafe { self.execution_engine.get_function("sum").ok() }
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
        let codegen = CodeGen {
            context: &context,
            module,
            builder: context.create_builder(),
            execution_engine,
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
                return Err("sparse jacobian not implemented")
            }
            else if options.jacobian == "dense" || options.jacobian == "none" {
                SUNDenseMatrix(number_of_states, number_of_states)
            }
            else if options.jacobian == "matrix-free" {
                null
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
                return Err("KLU linear solver not implemented")
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