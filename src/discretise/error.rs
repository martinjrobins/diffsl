use std::fmt;

use pest::Span;
use std::fmt::Write;

use crate::ast::StringSpan;

#[derive(Debug, Clone)]
pub struct ValidationError {
    text: String,
    source_ref: Option<StringSpan>,
}

impl ValidationError {
    pub fn new(text: String, span: Option<StringSpan>) -> Self {
        Self {
            text,
            source_ref: span,
        }
    }


    pub fn as_error_message(&self, f: &mut String, input: &str) -> fmt::Result {
        if let Some(source_ref) = self.source_ref {
            let span = Span::new(input, source_ref.pos_start, source_ref.pos_end);
            let (line, col) = span.as_ref().unwrap().start_pos().line_col();
            write!(f, "Line {}, Column {}: Error: {}", line, col, self.text)
        } else {
            write!(f, "Error: {}", self.text)
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(source_ref) = self.source_ref {
            write!(f, "{}: Error: {}", source_ref, self.text)
        } else {
            write!(f, "Error: {}", self.text)
        }
    }

}

#[derive(Debug, Clone)]
pub struct ValidationErrors {
    errors: Vec<ValidationError>,
}

impl ValidationErrors {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
        }
    }
    pub fn push(&mut self, new: ValidationError) {
        self.errors.push(new);
    }
    
    pub fn extend(&mut self, new: Vec<ValidationError>) {
        self.errors.extend(new)
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
    pub fn len(&self) -> usize {
        self.errors.len()
    }
    pub fn as_error_message(&self, input: &str) -> String {
        let mut buf = "\n".to_string();
        self.errors.iter().fold(Ok(()), |result, err| {
            result.and_then(|_| err.as_error_message(&mut buf, input)).and_then(|_| write!(buf, "\n"))
        }).unwrap();
        buf
    }
    
    pub fn has_error_contains(&self, text: &str) -> bool {
        self.errors.iter().any(|err| err.text.contains(text))
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.errors.iter().fold(Ok(()), |result, err| {
            result.and_then(|_| write!(f, "{}", err))
        })
    }
}
