use diffsl::execution::compiler::CompilerMode;
use diffsl::{Compiler, ExternalModule};

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
pub unsafe extern "C" fn set_u0(u: *mut f32, _data: *mut f32, _thread_id: u32, _thread_dim: u32) {
    if !u.is_null() {
        *u = 1.0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn rhs(
    _time: f32,
    u: *const f32,
    data: *mut f32,
    rr: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || rr.is_null() || data.is_null() {
        return;
    }
    let x = *u;
    let r = *data;
    *rr = r * x * (1.0 - x);
}

#[no_mangle]
pub unsafe extern "C" fn rhs_grad(
    _time: f32,
    u: *const f32,
    du: *const f32,
    data: *const f32,
    ddata: *mut f32,
    _rr: *const f32,
    drr: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || du.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
        return;
    }
    let x = *u;
    let dx = *du;
    let r = *data;
    *drr = r * (1.0 - 2.0 * x) * dx;
    *ddata = x * (1.0 - x);
}

#[no_mangle]
pub unsafe extern "C" fn rhs_rgrad(
    _time: f32,
    u: *const f32,
    du: *mut f32,
    data: *const f32,
    ddata: *mut f32,
    _rr: *const f32,
    drr: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || du.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
        return;
    }
    let x = *u;
    let r = *data;
    *du += r * (1.0 - 2.0 * x) * *drr;
    *ddata += x * (1.0 - x) * *drr;
}

#[no_mangle]
pub unsafe extern "C" fn rhs_sgrad(
    _time: f32,
    u: *const f32,
    data: *const f32,
    ddata: *mut f32,
    _rr: *const f32,
    drr: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
        return;
    }
    let x = *u;
    *drr = *data * x * (1.0 - x);
    *ddata = x * (1.0 - x);
}

#[no_mangle]
pub unsafe extern "C" fn rhs_srgrad(
    _time: f32,
    _u: *const f32,
    _data: *const f32,
    ddata: *mut f32,
    _rr: *const f32,
    drr: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !drr.is_null() {
        *drr = 0.0;
    }
    if !ddata.is_null() {
        *ddata = 0.0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn mass(
    _time: f32,
    v: *const f32,
    _data: *mut f32,
    mv: *mut f32,
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
    _time: f32,
    _v: *const f32,
    dv: *mut f32,
    _data: *const f32,
    _ddata: *mut f32,
    _mv: *const f32,
    dmv: *mut f32,
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
    _u: *const f32,
    _du: *mut f32,
    _data: *const f32,
    _ddata: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
}

#[no_mangle]
pub unsafe extern "C" fn set_u0_rgrad(
    _u: *const f32,
    _du: *mut f32,
    _data: *const f32,
    _ddata: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
}

#[no_mangle]
pub unsafe extern "C" fn set_u0_sgrad(
    _u: *const f32,
    _du: *mut f32,
    _data: *const f32,
    _ddata: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
}

#[no_mangle]
pub unsafe extern "C" fn calc_out(
    _time: f32,
    u: *const f32,
    _data: *mut f32,
    out: *mut f32,
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
    _time: f32,
    _u: *const f32,
    du: *const f32,
    _data: *const f32,
    ddata: *mut f32,
    _out: *const f32,
    dout: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if du.is_null() || ddata.is_null() || dout.is_null() {
        return;
    }
    *dout = *du;
    *ddata = 0.0;
}

#[no_mangle]
pub unsafe extern "C" fn calc_out_rgrad(
    _time: f32,
    _u: *const f32,
    du: *mut f32,
    _data: *const f32,
    _ddata: *mut f32,
    _out: *const f32,
    dout: *mut f32,
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
    _time: f32,
    _u: *const f32,
    _data: *const f32,
    ddata: *mut f32,
    _out: *const f32,
    dout: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !dout.is_null() {
        *dout = 0.0;
    }
    if !ddata.is_null() {
        *ddata = 0.0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn calc_out_srgrad(
    _time: f32,
    _u: *const f32,
    _data: *const f32,
    ddata: *mut f32,
    _out: *const f32,
    dout: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !dout.is_null() {
        *dout = 0.0;
    }
    if !ddata.is_null() {
        *ddata = 0.0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn calc_stop(
    _time: f32,
    u: *const f32,
    _data: *mut f32,
    root: *mut f32,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || root.is_null() {
        return;
    }
    *root = *u - 0.5;
}

#[no_mangle]
pub unsafe extern "C" fn set_id(id: *mut f32) {
    if !id.is_null() {
        *id = 42.0;
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
pub unsafe extern "C" fn set_inputs(inputs: *const f32, data: *mut f32) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    *data = *inputs;
}

#[no_mangle]
pub unsafe extern "C" fn get_inputs(inputs: *mut f32, data: *const f32) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    *inputs = *data;
}

#[no_mangle]
pub unsafe extern "C" fn set_inputs_grad(
    _inputs: *const f32,
    dinputs: *const f32,
    _data: *const f32,
    ddata: *mut f32,
) {
    if dinputs.is_null() || ddata.is_null() {
        return;
    }
    *ddata = *dinputs;
}

#[no_mangle]
pub unsafe extern "C" fn set_inputs_rgrad(
    _inputs: *const f32,
    dinputs: *mut f32,
    _data: *const f32,
    ddata: *mut f32,
) {
    if dinputs.is_null() || ddata.is_null() {
        return;
    }
    *dinputs += *ddata;
}

#[test]
fn external_module_compiler_runs_f32() {
    let module = ExternalModule::<f32>::new();
    let compiler = Compiler::from_codegen_module(module, CompilerMode::SingleThreaded)
        .expect("compiler should build");

    let (n_states, n_inputs, n_outputs, n_data, n_stop, has_mass) = compiler.get_dims();
    assert_eq!(n_states, STATES as usize);
    assert_eq!(n_inputs, INPUTS as usize);
    assert_eq!(n_outputs, OUTPUTS as usize);
    assert_eq!(n_data, DATA as usize);
    assert_eq!(n_stop, STOP as usize);
    assert!(has_mass);

    let mut data = vec![-1.0_f32; n_data];
    let inputs = vec![1.0_f32; n_inputs];
    compiler.set_inputs(&inputs, &mut data);

    let mut inputs_out = vec![-2.0_f32; n_inputs];
    compiler.get_inputs(&mut inputs_out, &data);
    assert_eq!(inputs_out, inputs);

    let mut u = vec![-2.0_f32; n_states];
    compiler.set_u0(&mut u, &mut data);
    assert_eq!(u[0], 1.0);

    let mut out = vec![-3.0_f32; n_outputs];
    compiler.calc_out(0.0, &u, &mut data, &mut out);
    assert_eq!(out[0], u[0]);

    let mut rr = vec![-4.0_f32; n_states];
    compiler.rhs(0.0, &u, &mut data, &mut rr);
    assert_eq!(rr[0], 0.0);

    let mut stop = vec![-5.0_f32; n_stop];
    compiler.calc_stop(0.0, &u, &mut data, &mut stop);
    assert_eq!(stop[0], 0.5);

    let mut mv = vec![-6.0_f32; n_states];
    compiler.mass(0.0, &u, &mut data, &mut mv);
    assert_eq!(mv[0], 1.0);

    let mut id = vec![-7.0_f32; n_states];
    compiler.set_id(&mut id);
    assert_eq!(id[0], 42.0);

    let du = vec![1.0_f32; n_states];
    let mut ddata = vec![-8.0_f32; n_data];
    let mut drr = vec![-9.0_f32; n_states];
    compiler.rhs_grad(0.0, &u, &du, &data, &mut ddata, &rr, &mut drr);
    assert_eq!(drr[0], -1.0);
    assert_eq!(ddata[0], 0.0);

    let mut dout = vec![-10.0_f32; n_outputs];
    compiler.calc_out_grad(0.0, &u, &du, &data, &mut ddata, &out, &mut dout);
    assert_eq!(dout[0], 1.0);
    assert_eq!(ddata[0], 0.0);

    let mut dinputs = vec![1.0_f32; n_inputs];
    compiler.set_inputs_grad(&inputs, &dinputs, &data, &mut ddata);
    assert_eq!(ddata[0], 1.0);

    let mut du_rev = vec![-11.0_f32; n_states];
    let mut ddata_rev = vec![-12.0_f32; n_data];
    let mut drr_rev = vec![1.0_f32; n_states];
    compiler.rhs_rgrad(
        0.0,
        &u,
        &mut du_rev,
        &data,
        &mut ddata_rev,
        &rr,
        &mut drr_rev,
    );
    assert_eq!(du_rev[0], -12.0);
    assert_eq!(ddata_rev[0], -12.0);

    let mut dv = vec![-13.0_f32; n_states];
    let mut dmv = vec![1.0_f32; n_states];
    compiler.mass_rgrad(0.0, &mut dv, &data, &mut ddata_rev, &mut dmv);
    assert_eq!(dv[0], -12.0);

    let mut dout_rev = vec![1.0_f32; n_outputs];
    compiler.calc_out_rgrad(
        0.0,
        &u,
        &mut du_rev,
        &data,
        &mut ddata_rev,
        &out,
        &mut dout_rev,
    );
    assert_eq!(du_rev[0], -11.0);

    compiler.set_inputs_rgrad(&inputs, &mut dinputs, &data, &mut ddata_rev);
    assert_eq!(dinputs[0], -11.0);

    let mut ddata_s = vec![-14.0_f32; n_data];
    let mut drr_s = vec![-15.0_f32; n_states];
    compiler.rhs_sgrad(0.0, &u, &data, &mut ddata_s, &rr, &mut drr_s);
    assert_eq!(drr_s[0], 0.0);
    assert_eq!(ddata_s[0], 0.0);

    let mut dout_s = vec![-16.0_f32; n_outputs];
    compiler.calc_out_sgrad(0.0, &u, &data, &mut ddata_s, &out, &mut dout_s);
    assert_eq!(dout_s[0], 0.0);

    let mut ddata_sr = vec![-17.0_f32; n_data];
    let mut drr_sr = vec![-18.0_f32; n_states];
    compiler.rhs_srgrad(0.0, &u, &data, &mut ddata_sr, &rr, &mut drr_sr);
    assert_eq!(drr_sr[0], 0.0);
    assert_eq!(ddata_sr[0], 0.0);

    let mut dout_sr = vec![-19.0_f32; n_outputs];
    compiler.calc_out_srgrad(0.0, &u, &data, &mut ddata_sr, &out, &mut dout_sr);
    assert_eq!(dout_sr[0], 0.0);
}
