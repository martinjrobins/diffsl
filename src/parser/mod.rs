use pest::error::Error;

pub mod ms_parser;
pub use ms_parser::MsParser;
pub use ms_parser::Rule as MsRule;

pub fn parse_ms_string(text: &str) -> Result<Vec<Box<Ast>>, Error<MsRule>> {
    ms_parser::parse_string(text)
}

pub mod ds_parser;
pub use ds_parser::DsParser;
pub use self::ds_parser::Rule as DsRule;

pub fn parse_ds_string(text: &str) -> Result<ast::DsModel, Error<DsRule>> {
    ds_parser::parse_string(text)
}

use crate::ast::{self, Ast};


#[cfg(test)]
mod tests {
    use std::fs;

    use pest::Parser;

    use super::{MsParser, MsRule};
    

    const MS_FILENAMES: &[&str] = &["test_circuit.ms", "test_fishers.ms", "test_pk.ms"];

    const BASE_DIR: &str = "src/parser";

    #[test]
    fn parse_examples() {
        for filename in MS_FILENAMES {
            let unparsed_file =
                fs::read_to_string(BASE_DIR.to_owned() + "/" + filename).expect("cannot read file");
            let _list = MsParser::parse(MsRule::main, &unparsed_file)
                .unwrap_or_else(|e| panic!("unsuccessful parse ({}) {}", filename, e));
        }
    }
}