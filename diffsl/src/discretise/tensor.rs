use std::fmt;

use ndarray::Array1;

use crate::ast::Ast;

use super::{ArcLayout, Layout, Shape};

pub type Index = Array1<i64>;

#[derive(Debug, Clone)]
// F(t, u, u_dot) = G(t, u)
pub struct TensorBlock<'s> {
    name: Option<String>,
    start: Index,
    indices: Vec<char>,
    layout: ArcLayout,
    expr_layout: ArcLayout,
    expr: Ast<'s>,
    tangent_expr: Ast<'s>,
}

impl<'s> TensorBlock<'s> {
    pub fn new(
        name: Option<String>,
        start: Index,
        indices: Vec<char>,
        layout: ArcLayout,
        expr_layout: ArcLayout,
        expr: Ast<'s>,
    ) -> Self {
        Self {
            name,
            start,
            indices,
            layout,
            expr_layout,
            tangent_expr: expr.tangent(),
            expr,
        }
    }
    pub fn new_dense_vector(name: Option<String>, start: i64, shape: usize, expr: Ast<'s>) -> Self {
        let layout = ArcLayout::new(Layout::dense(Shape::from_vec(vec![shape])));
        Self {
            name,
            start: Index::from_vec(vec![start]),
            layout: layout.clone(),
            expr_layout: layout,
            tangent_expr: expr.tangent(),
            expr,
            indices: Vec::new(),
        }
    }

    pub fn nnz(&self) -> usize {
        if self.is_diagonal() {
            self.shape().iter().product()
        } else {
            *self.shape().get(0).unwrap_or(&0usize)
        }
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn start(&self) -> &Index {
        &self.start
    }

    pub fn expr(&self) -> &Ast<'s> {
        &self.expr
    }

    pub fn tangent_expr(&self) -> &Ast<'s> {
        &self.tangent_expr
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn is_diagonal(&self) -> bool {
        self.layout.is_diagonal()
    }

    pub fn layout(&self) -> &ArcLayout {
        &self.layout
    }

    pub fn layout_ptr(&self) -> &ArcLayout {
        &self.layout
    }

    pub fn expr_layout(&self) -> &ArcLayout {
        &self.expr_layout
    }

    pub fn name(&self) -> Option<&str> {
        match &self.name {
            Some(name) => Some(name.as_str()),
            None => None,
        }
    }

    pub fn indices(&self) -> &[char] {
        self.indices.as_ref()
    }

    pub fn start_mut(&mut self) -> &mut Index {
        &mut self.start
    }
}

impl fmt::Display for TensorBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for i in 0..self.rank() {
            write!(f, "{}", self.start[i])?;
            if i < self.rank() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "){}: ", self.layout().as_ref())?;
        if let Some(name) = self.name.as_ref() {
            write!(f, "{} = ", name)?;
        }
        write!(f, "{} {}", self.expr, self.expr_layout().as_ref())
    }
}

#[derive(Debug, Clone)]
// F(t, u, u_dot) = G(t, u)
pub struct Tensor<'s> {
    name: &'s str,
    elmts: Vec<TensorBlock<'s>>,
    layout: ArcLayout,
    indices: Vec<char>,
}

impl<'s> Tensor<'s> {
    pub fn new_empty(name: &'s str) -> Self {
        Self {
            name,
            elmts: Vec::new(),
            indices: Vec::new(),
            layout: ArcLayout::new(Layout::dense(Shape::zeros(0))),
        }
    }

    pub fn is_dense(&self) -> bool {
        self.layout.is_dense()
    }

    pub fn is_same_layout(&self, other: &Self) -> bool {
        self.layout == other.layout
    }

    pub fn nnz(&self) -> usize {
        self.layout.nnz()
    }

    pub fn new(
        name: &'s str,
        elmts: Vec<TensorBlock<'s>>,
        layout: ArcLayout,
        indices: Vec<char>,
    ) -> Self {
        Self {
            name,
            elmts,
            indices,
            layout,
        }
    }

    pub fn new_no_layout(name: &'s str, elmts: Vec<TensorBlock<'s>>, indices: Vec<char>) -> Self {
        if elmts.is_empty() {
            Tensor::new("out", vec![], ArcLayout::new(Layout::new_empty(0)), vec![])
        } else {
            let layout = Layout::concatenate(elmts.as_slice(), indices.len()).unwrap();
            Tensor::new(name, elmts, ArcLayout::new(layout), indices)
        }
    }

    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn elmts(&self) -> &[TensorBlock<'s>] {
        self.elmts.as_ref()
    }

    pub fn indices(&self) -> &[char] {
        self.indices.as_ref()
    }

    pub fn layout_ptr(&self) -> &ArcLayout {
        &self.layout
    }

    pub fn layout(&self) -> &Layout {
        self.layout.as_ref()
    }

    pub fn set_name(&mut self, name: &'s str) {
        self.name = name;
    }
}

impl fmt::Display for Tensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.indices.is_empty() {
            write!(f, "{}_", self.name)?;
            for i in 0..self.indices.len() {
                write!(f, "{}", self.indices[i])?;
            }
        } else {
            write!(f, "{}", self.name)?;
        }
        writeln!(f, " {} {{", self.layout())?;
        for i in 0..self.elmts.len() {
            write!(f, "  {}", self.elmts[i])?;
            if i < self.elmts.len() - 1 {
                writeln!(f, ",")?;
            }
        }
        write!(f, "}}")
    }
}
