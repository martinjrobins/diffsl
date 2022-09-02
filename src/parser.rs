use crate::Rule;
use crate::pest::Parser;
use crate::MsParser;
use pest::error::Error;
use pest::iterators::Pair;

use itertools::Itertools;

use crate::ast::Ast;

//sign       = @{ ("-"|"+")? }
//factor_op  = @{ "*"|"/" }
fn parse_sign(pair: Pair<Rule>) -> char {
    print!("pair '{}'\n", pair.as_str());
    *pair
        .as_str()
        .chars()
        .collect::<Vec<char>>()
        .first()
        .unwrap()
}

//name       = @{ 'a'..'z' ~ ("_" | 'a'..'z' | 'A'..'Z' | '0'..'9')* }
//domain_name = @{ 'A'..'Z' ~ ('a'..'z' | 'A'..'Z' | '0'..'9')* }
fn parse_name(pair: Pair<Rule>) -> String {
    pair.as_str().to_string()
}

fn parse_value<'a>(pair: Pair<Rule>) -> Ast {
    match pair.as_rule() {
        // name       = @{ 'a'..'z' ~ ("_" | 'a'..'z' | 'A'..'Z' | '0'..'9')* }
        // domain_name = @{ 'A'..'Z' ~ ('a'..'z' | 'A'..'Z' | '0'..'9')* }
        Rule::name | Rule::domain_name => Ast::Name(pair.as_str().to_string()),

        // integer    = @{ ('0'..'9')+ }
        // real       = @{ ( ('0'..'9')+ ~ "." ~ ('0'..'9')+ ) | integer }
        Rule::integer | Rule::real => Ast::Name(pair.as_str().parse().unwrap()),

        // model = { "model" ~ name ~ "(" ~ unknown? ~ ("," ~ unknown)* ~ ")" ~ "{" ~ statement* ~ "}" }
        Rule::model => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let unknowns = inner
                .by_ref()
                .take_while_ref(|pair| pair.as_rule() == Rule::unknown)
                .map(parse_value)
                .map(Box::new)
                .collect();
            let statements = inner.map(parse_value).map(Box::new).collect();
            Ast::Model {
                name,
                unknowns,
                statements,
            }
        }
        // definition = { "let" ~ name ~ "=" ~ expression }
        Rule::definition => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            let rhs = Box::new(parse_value(inner.next().unwrap()));
            Ast::Definition { name, rhs }
        }

        // range      = { "[" ~ real ~ "..." ~ real ~ "]" }
        Rule::range => {
            let mut inner = pair.into_inner();
            Ast::Range {
                lower: inner.next().unwrap().as_str().parse().unwrap(),
                upper: inner.next().unwrap().as_str().parse().unwrap(),
            }
        }

        // domain     = { range | domain_name }
        Rule::domain => parse_value(pair.into_inner().next().unwrap()),

        // codomain   = { "->" ~ domain }
        Rule::codomain => parse_value(pair.into_inner().next().unwrap()),

        //unknown    = { name ~ dependents? ~ codomain? }
        Rule::unknown => {
            let mut inner = pair.into_inner();
            let name = parse_name(inner.next().unwrap());
            //dependents = { "(" ~ name ~ ("," ~ name )* ~ ")" }
            let dependents = if inner.peek().unwrap().as_rule() == Rule::dependents {
                inner.next().unwrap().into_inner().map(parse_name).collect()
            } else {
                Vec::new()
            };
            let codomain = if inner.peek().is_some() {
                Some(Box::new(parse_value(inner.next().unwrap())))
            } else {
                None
            };
            Ast::Unknown {
                name,
                dependents,
                codomain,
            }
        }
        //statement  = { definition | submodel | rate_equation | equation }
        Rule::statement => parse_value(pair.into_inner().next().unwrap()),

        //call_arg   = { (name ~ "=")? ~ expression }
        Rule::call_arg => {
            let mut inner = pair.into_inner();
            let name = if inner.peek().unwrap().as_rule() == Rule::name {
                Some(parse_name(inner.next().unwrap()))
            } else {
                None
            };
            Ast::CallArg {
                name,
                expression: Box::new(parse_value(inner.next().unwrap())),
            }
        }

        //call       = { name ~ "(" ~ call_arg ~ ("," ~ call_arg )* ~ ")" }
        Rule::call => {
            let mut inner = pair.into_inner();
            Ast::Call {
                fn_name: parse_name(inner.next().unwrap()),
                args: inner.map(parse_value).collect(),
            }
        }

        //submodel   = { "use" ~ call ~ ("as" ~ name)? }
        Rule::submodel => {
            // TODO: is there a better way of destructuring this?
            let mut inner = pair.into_inner();
            let (name, args) =
                if let Ast::Call { fn_name, args } = parse_value(inner.next().unwrap()) {
                    (fn_name, args)
                } else {
                    unreachable!()
                };
            let local_name = if inner.peek().is_some() {
                parse_name(inner.next().unwrap())
            } else {
                name.clone()
            };
            Ast::Submodel {
                name,
                local_name,
                args,
            }
        }

        //rate_equation = { "dot" ~ "(" ~ name ~ ")" ~ "+=" ~ expression }
        Rule::rate_equation => {
            let mut inner = pair.into_inner();
            Ast::RateEquation {
                name: parse_name(inner.next().unwrap()),
                rhs: Box::new(parse_value(inner.next().unwrap())),
            }
        }

        //equation   = { expression ~ "=" ~ expression }
        Rule::equation => {
            let mut inner = pair.into_inner();
            Ast::Equation {
                lhs: Box::new(parse_value(inner.next().unwrap())),
                rhs: Box::new(parse_value(inner.next().unwrap())),
            }
        }

        //expression = { sign? ~ term ~ (term_op ~ term)* }
        Rule::expression => {
            let mut inner = pair.into_inner();
            let sign = if inner.peek().unwrap().as_rule() == Rule::sign {
                Some(parse_sign(inner.next().unwrap()))
            } else {
                None
            };
            let mut head_term = parse_value(inner.next().unwrap());
            while inner.peek().is_some() {
                //term_op    = @{ "-"|"+" }
                let term_op = parse_sign(inner.next().unwrap());
                let rhs_term = parse_value(inner.next().unwrap());
                head_term = Ast::Binop {
                    op: term_op,
                    left: Box::new(head_term),
                    right: Box::new(rhs_term),
                };
            }
            if sign.is_some() {
                Ast::Monop {
                    op: sign.unwrap(),
                    child: Box::new(head_term),
                }
            } else {
                head_term
            }
        }

        //term       = { factor ~ (factor_op ~ factor)* }
        Rule::term => {
            let mut inner = pair.into_inner();
            let mut head_factor = parse_value(inner.next().unwrap());
            while inner.peek().is_some() {
                let factor_op = parse_sign(inner.next().unwrap());
                let rhs_factor = parse_value(inner.next().unwrap());
                head_factor = Ast::Binop {
                    op: factor_op,
                    left: Box::new(head_factor),
                    right: Box::new(rhs_factor),
                };
            }
            head_factor
        }

        // factor     = { call | name | real | integer | "(" ~ expression ~ ")" }
        Rule::factor => parse_value(pair.into_inner().next().unwrap()),

        _ => unreachable!("{:?}", pair.to_string()),
    }
}

pub fn parse_string(text: &str) -> Result<Vec<Ast>, Error<Rule>> {
    let main = MsParser::parse(Rule::main, &text)?.next().unwrap();
    let models = main
        .into_inner()
        .take_while(|pair| pair.as_rule() != Rule::EOI)
        .map(parse_value)
        .collect();
    return Ok(models);
}

#[cfg(test)]
mod tests {
    use super::parse_string;
    use crate::ast::Ast;

    #[test]
    fn empty_model() {
        const TEXT: &str = "model test() {}";
        let models = parse_string(TEXT).unwrap();
        assert_eq!(models.len(), 1);
        assert!(
            matches!(&models[0], Ast::Model { name, unknowns, statements } if name == "test" && unknowns.is_empty() && statements.is_empty()),
        );
    }

    #[test]
    fn capacitor_and_resistor_models() {
        let text = "
        model capacitor( i(t), v(t), c -> NonNegative) {
            i = c * dot(v)
        }
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        ";
        let models = parse_string(text).unwrap();
        assert_eq!(models.len(), 2);

        if let Ast::Model {
            name,
            unknowns,
            statements,
        } = &models[0]
        {
            assert_eq!(name, "capacitor");
            assert_eq!(unknowns.len(), 3);
            assert!(
                matches!(&*unknowns[0], Ast::Unknown { name, dependents, codomain } if name == "i" && dependents.len() == 1 && codomain.is_none())
            );
            assert!(
                matches!(&*unknowns[1], Ast::Unknown { name, dependents, codomain } if name == "v" && dependents.len() == 1 && codomain.is_none())
            );
            assert!(
                matches!(&*unknowns[2], Ast::Unknown { name, dependents, codomain } if name == "c" && dependents.len() == 0 && codomain.is_some())
            );
            assert_eq!(statements.len(), 1);
            if let Ast::Equation { lhs, rhs } = &*statements[0] {
                assert!(matches!(&**lhs, Ast::Name(name) if name == "i"));
                assert!(matches!(&**rhs, Ast::Binop{op, left: _, right: _} if *op == '*'));
            } else {
                assert!(false, "not an equation")
            }
        } else {
            assert!(false, "not a model");
        }
    }

    #[test]
    fn rate_equation() {
        let text = "
        model diffusion( x -> Omega, d -> NonNegative, y(x) ) { 
            dot(y) = d * div(grad(y, x), x) 
        }
        ";
        let models = parse_string(text).unwrap();
        assert_eq!(models.len(), 1);

        if let Ast::Model {
            name,
            unknowns,
            statements,
        } = &models[0]
        {
            assert_eq!(name, "diffusion");
            assert_eq!(unknowns.len(), 3);
            assert_eq!(statements.len(), 1);
            if let Ast::RateEquation { name, rhs } = &*statements[0] {
                assert_eq!(name, "y");
                assert!(matches!(&**rhs, Ast::Binop{op, left: _, right: _} if *op == '*'));
            } else {
                assert!(false, "not a rate equation")
            }
        } else {
            assert!(false, "not a model");
        }
    }

    #[test]
    fn submodel_and_let() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i1(t), i2(t), i3(t)) {
            let inputVoltage = sin(t) 
            use resistor(v = inputVoltage)
        }
        ";
        let models = parse_string(text).unwrap();
        assert_eq!(models.len(), 2);

        if let Ast::Model {
            name,
            unknowns,
            statements,
        } = &models[1]
        {
            assert_eq!(name, "circuit");
            assert_eq!(unknowns.len(), 3);
            assert_eq!(statements.len(), 2);
            if let Ast::Definition { name, rhs } = &*statements[0] {
                assert_eq!(name, "inputVoltage");
                assert!(
                    matches!(&**rhs, Ast::Call{fn_name, args} if fn_name == "sin" && args.len() == 1)
                );
            } else {
                assert!(false, "not an definition")
            }
            if let Ast::Submodel {
                name,
                local_name,
                args,
            } = &*statements[1]
            {
                assert_eq!(name, "resistor");
                assert_eq!(local_name, "resistor");
                assert_eq!(args.len(), 1);
                if let Ast::CallArg { name, expression } = &args[0] {
                    assert_eq!(name.as_ref().unwrap(), "v");
                    assert!(matches!(&**expression, Ast::Name(name) if name == "inputVoltage"));
                } else {
                    unreachable!("not a call arg")
                }
            } else {
                assert!(false, "not an definition")
            }
        } else {
            assert!(false, "not a model");
        }
    }
}
