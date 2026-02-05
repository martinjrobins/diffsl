use pest::error::Error;

pub mod ms_parser;
pub use ms_parser::MsParser;
pub use ms_parser::Rule as MsRule;

pub fn parse_ms_string(text: &str) -> Result<Vec<Ast<'_>>, Box<Error<MsRule>>> {
    ms_parser::parse_string(text)
}

pub mod ds_parser;
pub use self::ds_parser::Rule as DsRule;
pub use ds_parser::DsParser;

pub fn parse_ds_string(text: &str) -> Result<ast::DsModel<'_>, Box<Error<DsRule>>> {
    ds_parser::parse_string(text)
}

use crate::ast::{self, Ast};

#[cfg(test)]
mod tests {
    use pest::Parser;

    use super::{MsParser, MsRule};

    const MS_FILES: &[(&str, &str)] = &[
        ("test_circuit.ms", include_str!("test_circuit.ms")),
        ("test_fishers.ms", include_str!("test_fishers.ms")),
        ("test_pk.ms", include_str!("test_pk.ms")),
    ];

    #[test]
    fn parse_examples() {
        for (filename, contents) in MS_FILES {
            let _list = MsParser::parse(MsRule::main, contents)
                .unwrap_or_else(|e| panic!("unsuccessful parse ({filename}) {e}"));
        }
    }
}
