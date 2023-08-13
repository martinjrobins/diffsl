pub mod ms_parser;
pub use ms_parser::MsParser;

pub mod ds_parser;
pub use ds_parser::DsParser;

#[cfg(test)]
mod tests {
    use super::pest::Parser;
    use crate::ms_parser::MsParser;
    use crate::ms_parser::Rule;
    use std::fs;

    const MS_FILENAMES: &[&str] = &["test_circuit.ms", "test_fishers.ms", "test_pk.ms"];

    const BASE_DIR: &str = "src";

    #[test]
    fn parse_examples() {
        for filename in MS_FILENAMES {
            let unparsed_file =
                fs::read_to_string(BASE_DIR.to_owned() + "/" + filename).expect("cannot read file");
            let _list = MsParser::parse(Rule::main, &unparsed_file)
                .unwrap_or_else(|e| panic!("unsuccessful parse ({}) {}", filename, e));
        }
    }
}