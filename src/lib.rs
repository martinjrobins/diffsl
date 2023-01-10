extern crate pest;
#[macro_use]
extern crate pest_derive;

pub mod ms_parser;
pub mod ds_parser;
pub mod ast;
pub mod builder;
pub mod discretise;
pub mod codegen;
pub mod error;

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

