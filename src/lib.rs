extern crate pest;
#[macro_use]
extern crate pest_derive;

#[derive(Parser)]
#[grammar = "ms_grammar.pest"] // relative to src
struct MsParser;

#[cfg(test)]
mod tests {
    use crate::pest::Parser;
    use crate::MsParser;
    use crate::Rule;
    use std::fs;

    const MS_FILENAMES: &[&str] = &["test_circuit.ms", "test_fishers.ms", "test_pk.ms"];

    const BASE_DIR: &str = "src";

    #[test]
    fn can_parse() {
        for filename in MS_FILENAMES {
            let unparsed_file =
                fs::read_to_string(BASE_DIR.to_owned() + "/" + filename).expect("cannot read file");
            let _list =
                MsParser::parse(Rule::main, &unparsed_file).expect("unsuccessful parse");
            // unwrap the parse result
        }
    }
}
