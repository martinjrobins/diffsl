include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../support/external_test_macros.rs"
));

define_external_test!(f32, external_dynamic_fixture_symbols_f32);
