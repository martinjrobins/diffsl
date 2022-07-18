extern crate pest;
#[macro_use]
extern crate pest_derive;

#[derive(Parser)]
#[grammar = "mmt_grammar.pest"] // relative to src
struct MmtParser;

#[cfg(test)]
mod tests {
    use crate::pest::Parser;
    use crate::MmtParser;
    use crate::Rule;
    use std::fs;

    #[test]
    fn it_works() {
        let unparsed_file = fs::read_to_string("src/test.mmt").expect("cannot read file");
        let list = MmtParser::parse(Rule::ident_list, &unparsed_file)
            .expect("unsuccessful parse") // unwrap the parse result
            .next()
            .unwrap(); // get and unwrap the `file` rule; never fails

        let mut record_count: u64 = 0;
        for record in list.into_inner() {
            match record.as_rule() {
                Rule::ident => {
                    record_count += 1;
                }
                _ => unreachable!(),
            }
        }

        println!("Number of records: {}", record_count);
    }


}
