#[cfg(all(feature = "external_dynamic", not(target_arch = "wasm32")))]
include!("support/external_dynamic_test_macros.rs");

#[cfg(all(feature = "external_dynamic", not(target_arch = "wasm32")))]
#[test]
fn external_dynamic_module_missing_required_symbol_errors() {
    let fixture_path = build_fixture_library("external_dynamic_fixture_missing_rhs");
    let err = diffsl::ExternalDynModule::<f64>::new(&fixture_path)
        .err()
        .expect("module construction should fail when rhs is missing");
    assert!(err.to_string().contains("Missing required symbol:"));
}
