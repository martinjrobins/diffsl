use std::{fmt, error};

#[derive(Debug, Clone)]
struct Error {
    msg: String,
}

impl fmt::Display for Error{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        &self.msg
    }
}

impl From<String> for Error {
    fn from(error: String) -> Self {
        Error { msg: error }
    }
}

impl From<&str> for Error {
    fn from(error: &str) -> Self {
        Error { msg: error.to_owned() }
    }
}
