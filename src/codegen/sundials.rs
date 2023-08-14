use inkwell::{execution_engine::JitFunction, passes::PassManager, OptimizationLevel};
use ndarray::{Array1, s, Array2, ShapeBuilder};
use sundials_sys::{realtype, N_Vector, IDAGetNonlinSolvStats, IDA_SUCCESS, IDA_ROOT_RETURN, IDA_YA_YDP_INIT, IDA_NORMAL, IDASolve, IDAGetIntegratorStats, IDASetStopTime, IDACreate, N_VNew_Serial, N_VGetArrayPointer, N_VConst, IDAInit, IDACalcIC, IDASVtolerances, IDASetUserData, SUNLinSolInitialize, IDASetId, SUNMatrix, SUNLinearSolver, SUNDenseMatrix, PREC_NONE, PREC_LEFT, SUNLinSol_Dense, SUNLinSol_SPBCGS, SUNLinSol_SPFGMR, SUNLinSol_SPGMR, SUNLinSol_SPTFQMR, IDASetLinearSolver, SUNLinSolFree, SUNMatDestroy, N_VDestroy, IDAFree, IDAReInit, IDAGetConsistentIC, IDAGetReturnFlagName};
use std::{ffi::{c_void, CStr, c_int}, io::{self, Write}, collections::HashMap, iter::zip, ptr::null_mut};
use anyhow::{anyhow, Result};

use crate::discretise::{DiscreteModel, Layout};

use super::{DataLayout, codegen::{ResidualFunc, U0Func, CalcOutFunc}, CodeGen};
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
        let mut codegen = CodeGen::new(model, &context, module, fpm, ee, real_type, real_type_str);

        let set_u0 = codegen.compile_set_u0(model)?;
        let residual = codegen.compile_residual(model)?;
        let calc_out = codegen.compile_calc_out(model)?;

        set_u0.print_to_stderr();
        residual.print_to_stderr();
        calc_out.print_to_stderr();

        let set_u0 = codegen.jit::<U0Func>(set_u0)?;
        let residual = codegen.jit::<ResidualFunc>(residual)?;
        let calc_out = codegen.jit::<CalcOutFunc>(calc_out)?;

        let data_layout = DataLayout::new(model);
        
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