pub mod ms_parser;
use std::error::Error;

pub use ms_parser::MsParser;

pub fn parse_ds_string(text: &str) -> Result<ast::DsModel, Error<Rule>> {
    ds_parser::parse_string(text)
}

pub mod ds_parser;
pub use ds_parser::DsParser;

pub fn parse_ms_string(text: &str) -> Result<ast::MsModel, Error<Rule>> {
    ms_parser::parse_string(text)
}

use crate::ast;

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