#[no_mangle]
pub unsafe extern "C" fn set_u0(u: *mut f64, _data: *mut f64, _thread_id: u32, _thread_dim: u32) {
    if !u.is_null() {
        *u = 1.0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn reset(
    _time: f64,
    _u: *const f64,
    _data: *mut f64,
    reset: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !reset.is_null() {
        *reset = 0.0;
    }
}
