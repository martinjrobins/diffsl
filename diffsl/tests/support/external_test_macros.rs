use diffsl::execution::compiler::CompilerMode;
use diffsl::{Compiler, ExternalModule};

macro_rules! define_external_test {
    ($ty:ty, $test_name:ident) => {
        const STATES: u32 = 1;
        const INPUTS: u32 = 1;
        const OUTPUTS: u32 = 1;
        const DATA: u32 = 1;
        const STOP: u32 = 1;

        #[no_mangle]
        pub unsafe extern "C" fn barrier_init() {}

        #[no_mangle]
        pub unsafe extern "C" fn set_constants(_thread_id: u32, _thread_dim: u32) {}

        #[no_mangle]
        pub unsafe extern "C" fn set_u0(
            u: *mut $ty,
            _data: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if !u.is_null() {
                *u = 1.0 as $ty;
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn rhs(
            _time: $ty,
            u: *const $ty,
            data: *mut $ty,
            rr: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if u.is_null() || rr.is_null() || data.is_null() {
                return;
            }
            let x = *u;
            let r = *data;
            *rr = r * x * ((1.0 as $ty) - x);
        }

        #[no_mangle]
        pub unsafe extern "C" fn rhs_grad(
            _time: $ty,
            u: *const $ty,
            du: *const $ty,
            data: *const $ty,
            ddata: *mut $ty,
            _rr: *const $ty,
            drr: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if u.is_null() || du.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
                return;
            }
            let x = *u;
            let dx = *du;
            let r = *data;
            *drr = r * ((1.0 as $ty) - (2.0 as $ty) * x) * dx;
            *ddata = x * ((1.0 as $ty) - x);
        }

        #[no_mangle]
        pub unsafe extern "C" fn rhs_rgrad(
            _time: $ty,
            u: *const $ty,
            du: *mut $ty,
            data: *const $ty,
            ddata: *mut $ty,
            _rr: *const $ty,
            drr: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if u.is_null() || du.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
                return;
            }
            let x = *u;
            let r = *data;
            *du += r * ((1.0 as $ty) - (2.0 as $ty) * x) * *drr;
            *ddata += x * ((1.0 as $ty) - x) * *drr;
        }

        #[no_mangle]
        pub unsafe extern "C" fn rhs_sgrad(
            _time: $ty,
            u: *const $ty,
            data: *const $ty,
            ddata: *mut $ty,
            _rr: *const $ty,
            drr: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if u.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
                return;
            }
            let x = *u;
            *drr = *data * x * ((1.0 as $ty) - x);
            *ddata = x * ((1.0 as $ty) - x);
        }

        #[no_mangle]
        pub unsafe extern "C" fn rhs_srgrad(
            _time: $ty,
            _u: *const $ty,
            _data: *const $ty,
            ddata: *mut $ty,
            _rr: *const $ty,
            drr: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if !drr.is_null() {
                *drr = 0.0 as $ty;
            }
            if !ddata.is_null() {
                *ddata = 0.0 as $ty;
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn mass(
            _time: $ty,
            v: *const $ty,
            _data: *mut $ty,
            mv: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if v.is_null() || mv.is_null() {
                return;
            }
            *mv = *v;
        }

        #[no_mangle]
        pub unsafe extern "C" fn mass_rgrad(
            _time: $ty,
            _v: *const $ty,
            dv: *mut $ty,
            _data: *const $ty,
            _ddata: *mut $ty,
            _mv: *const $ty,
            dmv: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if dv.is_null() || dmv.is_null() {
                return;
            }
            *dv += *dmv;
        }

        #[no_mangle]
        pub unsafe extern "C" fn set_u0_grad(
            _u: *const $ty,
            _du: *mut $ty,
            _data: *const $ty,
            _ddata: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
        }

        #[no_mangle]
        pub unsafe extern "C" fn set_u0_rgrad(
            _u: *const $ty,
            _du: *mut $ty,
            _data: *const $ty,
            _ddata: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
        }

        #[no_mangle]
        pub unsafe extern "C" fn set_u0_sgrad(
            _u: *const $ty,
            _du: *mut $ty,
            _data: *const $ty,
            _ddata: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
        }

        #[no_mangle]
        pub unsafe extern "C" fn calc_out(
            _time: $ty,
            u: *const $ty,
            _data: *mut $ty,
            out: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if u.is_null() || out.is_null() {
                return;
            }
            *out = *u;
        }

        #[no_mangle]
        pub unsafe extern "C" fn calc_out_grad(
            _time: $ty,
            _u: *const $ty,
            du: *const $ty,
            _data: *const $ty,
            ddata: *mut $ty,
            _out: *const $ty,
            dout: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if du.is_null() || ddata.is_null() || dout.is_null() {
                return;
            }
            *dout = *du;
            *ddata = 0.0 as $ty;
        }

        #[no_mangle]
        pub unsafe extern "C" fn calc_out_rgrad(
            _time: $ty,
            _u: *const $ty,
            du: *mut $ty,
            _data: *const $ty,
            _ddata: *mut $ty,
            _out: *const $ty,
            dout: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if du.is_null() || dout.is_null() {
                return;
            }
            *du += *dout;
        }

        #[no_mangle]
        pub unsafe extern "C" fn calc_out_sgrad(
            _time: $ty,
            _u: *const $ty,
            _data: *const $ty,
            ddata: *mut $ty,
            _out: *const $ty,
            dout: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if !dout.is_null() {
                *dout = 0.0 as $ty;
            }
            if !ddata.is_null() {
                *ddata = 0.0 as $ty;
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn calc_out_srgrad(
            _time: $ty,
            _u: *const $ty,
            _data: *const $ty,
            ddata: *mut $ty,
            _out: *const $ty,
            dout: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if !dout.is_null() {
                *dout = 0.0 as $ty;
            }
            if !ddata.is_null() {
                *ddata = 0.0 as $ty;
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn calc_stop(
            _time: $ty,
            u: *const $ty,
            _data: *mut $ty,
            root: *mut $ty,
            _thread_id: u32,
            _thread_dim: u32,
        ) {
            if u.is_null() || root.is_null() {
                return;
            }
            *root = *u - (0.5 as $ty);
        }

        #[no_mangle]
        pub unsafe extern "C" fn set_id(id: *mut $ty) {
            if !id.is_null() {
                *id = 42.0 as $ty;
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn get_dims(
            states: *mut u32,
            inputs: *mut u32,
            outputs: *mut u32,
            data: *mut u32,
            stop: *mut u32,
            has_mass: *mut u32,
        ) {
            if !states.is_null() {
                *states = STATES;
            }
            if !inputs.is_null() {
                *inputs = INPUTS;
            }
            if !outputs.is_null() {
                *outputs = OUTPUTS;
            }
            if !data.is_null() {
                *data = DATA;
            }
            if !stop.is_null() {
                *stop = STOP;
            }
            if !has_mass.is_null() {
                *has_mass = 1;
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn set_inputs(inputs: *const $ty, data: *mut $ty) {
            if inputs.is_null() || data.is_null() {
                return;
            }
            *data = *inputs;
        }

        #[no_mangle]
        pub unsafe extern "C" fn get_inputs(inputs: *mut $ty, data: *const $ty) {
            if inputs.is_null() || data.is_null() {
                return;
            }
            *inputs = *data;
        }

        #[no_mangle]
        pub unsafe extern "C" fn set_inputs_grad(
            _inputs: *const $ty,
            dinputs: *const $ty,
            _data: *const $ty,
            ddata: *mut $ty,
        ) {
            if dinputs.is_null() || ddata.is_null() {
                return;
            }
            *ddata = *dinputs;
        }

        #[no_mangle]
        pub unsafe extern "C" fn set_inputs_rgrad(
            _inputs: *const $ty,
            dinputs: *mut $ty,
            _data: *const $ty,
            ddata: *mut $ty,
        ) {
            if dinputs.is_null() || ddata.is_null() {
                return;
            }
            *dinputs += *ddata;
        }

        #[test]
        fn $test_name() {
            let module = ExternalModule::<$ty>::new();
            let compiler = Compiler::from_codegen_module(module, CompilerMode::SingleThreaded)
                .expect("compiler should build");

            let (n_states, n_inputs, n_outputs, n_data, n_stop, has_mass) = compiler.get_dims();
            assert_eq!(n_states, STATES as usize);
            assert_eq!(n_inputs, INPUTS as usize);
            assert_eq!(n_outputs, OUTPUTS as usize);
            assert_eq!(n_data, DATA as usize);
            assert_eq!(n_stop, STOP as usize);
            assert!(has_mass);

            let mut data = vec![-1.0 as $ty; n_data];
            let inputs = vec![1.0 as $ty; n_inputs];
            compiler.set_inputs(&inputs, &mut data);

            let mut inputs_out = vec![-2.0 as $ty; n_inputs];
            compiler.get_inputs(&mut inputs_out, &data);
            assert_eq!(inputs_out, inputs);

            let mut u = vec![-2.0 as $ty; n_states];
            compiler.set_u0(&mut u, &mut data);
            assert_eq!(u[0], 1.0 as $ty);

            let mut out = vec![-3.0 as $ty; n_outputs];
            compiler.calc_out(0.0 as $ty, &u, &mut data, &mut out);
            assert_eq!(out[0], u[0]);

            let mut rr = vec![-4.0 as $ty; n_states];
            compiler.rhs(0.0 as $ty, &u, &mut data, &mut rr);
            assert_eq!(rr[0], 0.0 as $ty);

            let mut stop = vec![-5.0 as $ty; n_stop];
            compiler.calc_stop(0.0 as $ty, &u, &mut data, &mut stop);
            assert_eq!(stop[0], 0.5 as $ty);

            let mut mv = vec![-6.0 as $ty; n_states];
            compiler.mass(0.0 as $ty, &u, &mut data, &mut mv);
            assert_eq!(mv[0], 1.0 as $ty);

            let mut id = vec![-7.0 as $ty; n_states];
            compiler.set_id(&mut id);
            assert_eq!(id[0], 42.0 as $ty);

            let du = vec![1.0 as $ty; n_states];
            let mut ddata = vec![-8.0 as $ty; n_data];
            let mut drr = vec![-9.0 as $ty; n_states];
            compiler.rhs_grad(0.0 as $ty, &u, &du, &data, &mut ddata, &rr, &mut drr);
            assert_eq!(drr[0], -1.0 as $ty);
            assert_eq!(ddata[0], 0.0 as $ty);

            let mut dout = vec![-10.0 as $ty; n_outputs];
            compiler.calc_out_grad(0.0 as $ty, &u, &du, &data, &mut ddata, &out, &mut dout);
            assert_eq!(dout[0], 1.0 as $ty);
            assert_eq!(ddata[0], 0.0 as $ty);

            let mut dinputs = vec![1.0 as $ty; n_inputs];
            compiler.set_inputs_grad(&inputs, &dinputs, &data, &mut ddata);
            assert_eq!(ddata[0], 1.0 as $ty);

            let mut du_rev = vec![-11.0 as $ty; n_states];
            let mut ddata_rev = vec![-12.0 as $ty; n_data];
            let mut drr_rev = vec![1.0 as $ty; n_states];
            compiler.rhs_rgrad(
                0.0 as $ty,
                &u,
                &mut du_rev,
                &data,
                &mut ddata_rev,
                &rr,
                &mut drr_rev,
            );
            assert_eq!(du_rev[0], -12.0 as $ty);
            assert_eq!(ddata_rev[0], -12.0 as $ty);

            let mut dv = vec![-13.0 as $ty; n_states];
            let mut dmv = vec![1.0 as $ty; n_states];
            compiler.mass_rgrad(0.0 as $ty, &mut dv, &data, &mut ddata_rev, &mut dmv);
            assert_eq!(dv[0], -12.0 as $ty);

            let mut dout_rev = vec![1.0 as $ty; n_outputs];
            compiler.calc_out_rgrad(
                0.0 as $ty,
                &u,
                &mut du_rev,
                &data,
                &mut ddata_rev,
                &out,
                &mut dout_rev,
            );
            assert_eq!(du_rev[0], -11.0 as $ty);

            compiler.set_inputs_rgrad(&inputs, &mut dinputs, &data, &mut ddata_rev);
            assert_eq!(dinputs[0], -11.0 as $ty);

            let mut ddata_s = vec![-14.0 as $ty; n_data];
            let mut drr_s = vec![-15.0 as $ty; n_states];
            compiler.rhs_sgrad(0.0 as $ty, &u, &data, &mut ddata_s, &rr, &mut drr_s);
            assert_eq!(drr_s[0], 0.0 as $ty);
            assert_eq!(ddata_s[0], 0.0 as $ty);

            let mut dout_s = vec![-16.0 as $ty; n_outputs];
            compiler.calc_out_sgrad(0.0 as $ty, &u, &data, &mut ddata_s, &out, &mut dout_s);
            assert_eq!(dout_s[0], 0.0 as $ty);

            let mut ddata_sr = vec![-17.0 as $ty; n_data];
            let mut drr_sr = vec![-18.0 as $ty; n_states];
            compiler.rhs_srgrad(0.0 as $ty, &u, &data, &mut ddata_sr, &rr, &mut drr_sr);
            assert_eq!(drr_sr[0], 0.0 as $ty);
            assert_eq!(ddata_sr[0], 0.0 as $ty);

            let mut dout_sr = vec![-19.0 as $ty; n_outputs];
            compiler.calc_out_srgrad(0.0 as $ty, &u, &data, &mut ddata_sr, &out, &mut dout_sr);
            assert_eq!(dout_sr[0], 0.0 as $ty);
        }
    };
}
