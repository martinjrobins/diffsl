use super::UIntType;

#[allow(clashing_extern_declarations)]
extern "C" {
    #[link_name = "barrier_init"]
    pub fn barrier_init_f32();
    #[link_name = "set_constants"]
    pub fn set_constants_f32(thread_id: UIntType, thread_dim: UIntType);
    #[link_name = "set_u0"]
    pub fn set_u0_f32(u: *mut f32, data: *mut f32, thread_id: UIntType, thread_dim: UIntType);
    #[link_name = "rhs"]
    pub fn rhs_f32(
        time: f32,
        u: *const f32,
        data: *mut f32,
        rr: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_grad"]
    pub fn rhs_grad_f32(
        time: f32,
        u: *const f32,
        du: *const f32,
        data: *const f32,
        ddata: *mut f32,
        rr: *const f32,
        drr: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_rgrad"]
    pub fn rhs_rgrad_f32(
        time: f32,
        u: *const f32,
        du: *mut f32,
        data: *const f32,
        ddata: *mut f32,
        rr: *const f32,
        drr: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_sgrad"]
    pub fn rhs_sgrad_f32(
        time: f32,
        u: *const f32,
        data: *const f32,
        ddata: *mut f32,
        rr: *const f32,
        drr: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "rhs_srgrad"]
    pub fn rhs_srgrad_f32(
        time: f32,
        u: *const f32,
        data: *const f32,
        ddata: *mut f32,
        rr: *const f32,
        drr: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "mass"]
    pub fn mass_f32(
        time: f32,
        v: *const f32,
        data: *mut f32,
        mv: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "mass_rgrad"]
    pub fn mass_rgrad_f32(
        time: f32,
        v: *const f32,
        dv: *mut f32,
        data: *const f32,
        ddata: *mut f32,
        mv: *const f32,
        dmv: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_u0_grad"]
    pub fn set_u0_grad_f32(
        u: *const f32,
        du: *mut f32,
        data: *const f32,
        ddata: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_u0_rgrad"]
    pub fn set_u0_rgrad_f32(
        u: *const f32,
        du: *mut f32,
        data: *const f32,
        ddata: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_u0_sgrad"]
    pub fn set_u0_sgrad_f32(
        u: *const f32,
        du: *mut f32,
        data: *const f32,
        ddata: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out"]
    pub fn calc_out_f32(
        time: f32,
        u: *const f32,
        data: *mut f32,
        out: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_grad"]
    pub fn calc_out_grad_f32(
        time: f32,
        u: *const f32,
        du: *const f32,
        data: *const f32,
        ddata: *mut f32,
        out: *const f32,
        dout: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_rgrad"]
    pub fn calc_out_rgrad_f32(
        time: f32,
        u: *const f32,
        du: *mut f32,
        data: *const f32,
        ddata: *mut f32,
        out: *const f32,
        dout: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_sgrad"]
    pub fn calc_out_sgrad_f32(
        time: f32,
        u: *const f32,
        data: *const f32,
        ddata: *mut f32,
        out: *const f32,
        dout: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_out_srgrad"]
    pub fn calc_out_srgrad_f32(
        time: f32,
        u: *const f32,
        data: *const f32,
        ddata: *mut f32,
        out: *const f32,
        dout: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "calc_stop"]
    pub fn calc_stop_f32(
        time: f32,
        u: *const f32,
        data: *mut f32,
        root: *mut f32,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    #[link_name = "set_id"]
    pub fn set_id_f32(id: *mut f32);
    #[link_name = "get_dims"]
    pub fn get_dims_f32(
        states: *mut UIntType,
        inputs: *mut UIntType,
        outputs: *mut UIntType,
        data: *mut UIntType,
        stop: *mut UIntType,
        has_mass: *mut UIntType,
    );
    #[link_name = "set_inputs"]
    pub fn set_inputs_f32(inputs: *const f32, data: *mut f32);
    #[link_name = "get_inputs"]
    pub fn get_inputs_f32(inputs: *mut f32, data: *const f32);
    #[link_name = "set_inputs_grad"]
    pub fn set_inputs_grad_f32(
        inputs: *const f32,
        dinputs: *const f32,
        data: *const f32,
        ddata: *mut f32,
    );
    #[link_name = "set_inputs_rgrad"]
    pub fn set_inputs_rgrad_f32(
        inputs: *const f32,
        dinputs: *mut f32,
        data: *const f32,
        ddata: *mut f32,
    );
}
