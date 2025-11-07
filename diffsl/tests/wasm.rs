use wasm_bindgen_test::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen_test]
async fn basic_test() {
    // read wasm from file model.wasm
    println!("Running wasm basic_test...");
    use diffsl::execution::{interface::RhsFunc, wasm::WasmJitModule, module::CodegenModuleJit};
    let wasm_bytes = include_bytes!("../../diffsl-backend/model.wasm");
    let mut module = WasmJitModule::new(wasm_bytes).await;
    let symbols = module.jit().unwrap();
    let rhs = symbols.get("rhs").unwrap();
    let rhs_func = unsafe {
        std::mem::transmute::<*const u8, RhsFunc>( *rhs)
    };
    let t= 1.0;
    let u = [0.1f64, 0.2f64]; // [y, z]
    let mut data = vec![0.3f64, 0.4f64]; // [r, k]
    let mut rr = vec![0.0f64, 0.0f64]; // [dy/dt, dz/dt]
                                       
    unsafe { rhs_func(t, u.as_ptr(), data.as_mut_ptr(), rr.as_mut_ptr(), 0, 1) };
    //F_i {
    //    (r * y) * (1 - (y / k)),
    //    (2 * y) - z,
    //}
    assert!((rr[0] - (data[0] * u[0] * (1.0 - (u[0] / data[1])))).abs() < 1e-10);
    assert!((rr[1] - ((2.0 * u[0]) - u[1])).abs() < 1e-10);
}