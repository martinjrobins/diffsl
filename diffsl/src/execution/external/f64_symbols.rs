use super::UIntType;

#[allow(clashing_extern_declarations)]
extern "C" {
    #[link_name = "barrier_init"]
    pub fn barrier_init_f64();
    #[link_name = "set_constants"]
    pub fn set_constants_f64(thread_id: UIntType, thread_dim: UIntType);
    #[link_name = "set_u0"]
    pub fn set_u0_f64(u: *mut f64, data: *mut f64, thread_id: UIntType, thread_dim: UIntType);
    #[link_name = "rhs"]
    pub fn rhs_f64(
        time: f64,
        u: *const f64,
        data: *mut f64,
        rr: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_grad"]
    pub fn rhs_grad_f64(
        time: f64,
        u: *const f64,
        du: *const f64,
        data: *const f64,
        ddata: *mut f64,
        rr: *const f64,
        drr: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_rgrad"]
    pub fn rhs_rgrad_f64(
        time: f64,
        u: *const f64,
        du: *mut f64,
        data: *const f64,
        ddata: *mut f64,
        rr: *const f64,
        drr: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_sgrad"]
    pub fn rhs_sgrad_f64(
        time: f64,
        u: *const f64,
        data: *const f64,
        ddata: *mut f64,
        rr: *const f64,
        drr: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_srgrad"]
    pub fn rhs_srgrad_f64(
        time: f64,
        u: *const f64,
        data: *const f64,
        ddata: *mut f64,
        rr: *const f64,
        drr: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "mass"]
    pub fn mass_f64(
        time: f64,
        v: *const f64,
        data: *mut f64,
        mv: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "mass_rgrad"]
    pub fn mass_rgrad_f64(
        time: f64,
        v: *const f64,
        dv: *mut f64,
        data: *const f64,
        ddata: *mut f64,
        mv: *const f64,
        dmv: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_u0_grad"]
    pub fn set_u0_grad_f64(
        u: *const f64,
        du: *mut f64,
        data: *const f64,
        ddata: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_u0_rgrad"]
    pub fn set_u0_rgrad_f64(
        u: *const f64,
        du: *mut f64,
        data: *const f64,
        ddata: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_u0_sgrad"]
    pub fn set_u0_sgrad_f64(
        u: *const f64,
        du: *mut f64,
        data: *const f64,
        ddata: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out"]
    pub fn calc_out_f64(
        time: f64,
        u: *const f64,
        data: *mut f64,
        out: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_grad"]
    pub fn calc_out_grad_f64(
        time: f64,
        u: *const f64,
        du: *const f64,
        data: *const f64,
        ddata: *mut f64,
        out: *const f64,
        dout: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_rgrad"]
    pub fn calc_out_rgrad_f64(
        time: f64,
        u: *const f64,
        du: *mut f64,
        data: *const f64,
        ddata: *mut f64,
        out: *const f64,
        dout: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_sgrad"]
    pub fn calc_out_sgrad_f64(
        time: f64,
        u: *const f64,
        data: *const f64,
        ddata: *mut f64,
        out: *const f64,
        dout: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_srgrad"]
    pub fn calc_out_srgrad_f64(
        time: f64,
        u: *const f64,
        data: *const f64,
        ddata: *mut f64,
        out: *const f64,
        dout: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_stop"]
    pub fn calc_stop_f64(
        time: f64,
        u: *const f64,
        data: *mut f64,
        root: *mut f64,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_id"]
    pub fn set_id_f64(id: *mut f64);
    #[link_name = "get_dims"]
    pub fn get_dims_f64(
        states: *mut UIntType,
        inputs: *mut UIntType,
        outputs: *mut UIntType,
        data: *mut UIntType,
        stop: *mut UIntType,
        has_mass: *mut UIntType,
    );
    #[link_name = "set_inputs"]
    pub fn set_inputs_f64(inputs: *const f64, data: *mut f64);
    #[link_name = "get_inputs"]
    pub fn get_inputs_f64(inputs: *mut f64, data: *const f64);
    #[link_name = "set_inputs_grad"]
    pub fn set_inputs_grad_f64(
        inputs: *const f64,
        dinputs: *const f64,
        data: *const f64,
        ddata: *mut f64,
    );
    #[link_name = "set_inputs_rgrad"]
    pub fn set_inputs_rgrad_f64(
        inputs: *const f64,
        dinputs: *mut f64,
        data: *const f64,
        ddata: *mut f64,
    );
}
