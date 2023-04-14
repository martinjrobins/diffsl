use anyhow::Result;
use anyhow::anyhow;
use ndarray::s;
use core::panic;
use std::cell::RefCell;
use std::ops::Deref;
use std::hash::Hash;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hasher;
use std::rc::Rc;

use itertools::chain;
use ndarray::Array1;

use crate::ast;
use crate::ast::Ast;
use crate::ast::AstKind;
use crate::ast::Call;
use crate::ast::Indice;
use crate::ast::TensorElmt;
use crate::ast::StringSpan;
use crate::builder::ModelInfo;
use crate::builder::Variable;
use crate::error::ValidationError;
use crate::error::ValidationErrors;

#[derive(Debug, Clone, PartialEq)]
enum LayoutKind {
    Dense,
    Diagonal,
    Sparse,
}


// a sparsity pattern for a multidimensional tensor. A tensor can be sparse, diagonal or dense, as given by the kind field.
// A tensor can also have n_dense_axes axes which are dense, these are the last n_dense_axes axes of the tensor. So for example,
// you could have a 2D sparse tensor with 1 dense axis, combining to give a tensor of rank 3 (i.e. the shape is length 3).
// indices are kept in row major order, so the last index is the fastest changing index.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    indices: Vec<Index>,
    shape: Shape,
    kind: LayoutKind,
    n_dense_axes: usize,
}

impl Layout {
    
    fn unravel_index(index: usize, shape: &Shape) -> Index {
        let mut idx = index;
        let mut res = Index::zeros(shape.len());
        for i in 0..shape.len() {
            res[i] = i64::try_from(idx % shape[i]).unwrap();
            idx = idx / shape[i];
        }
        res
    }

    // contract_last_axis contracts the last axis of the layout, returning a new layout with the last axis contracted.
    fn contract_last_axis(&self) -> Result<Layout> {
        let rank = self.rank();
        if rank == 0 {
            return Err(anyhow!("cannot contract last axis of a scalar"));
        }
        let new_shape = self.shape.slice(s![0..rank-1]).to_owned();
        let mut new_indices = self.indices.clone();

        // if the layout is sparse and there are no dense axes, then remove the last axis from each index
        if self.is_sparse() && self.n_dense_axes == 0 {
            for i in 0..self.indices.len() {
                new_indices[i] = new_indices[i].slice(s![0..rank-1]).to_owned();
            }
        } 
        new_indices.pop();
        Ok(Layout {
            indices: new_indices,
            shape: new_shape,
            kind: self.kind.clone(),
            n_dense_axes: self.n_dense_axes,
        })
    }


    // permute the axes of the layout and return a new layout
    fn permute(&self, permutation: &[usize]) -> Result<Layout> {
        let rank = self.rank();
        // check that permutation is a valid permutation
        if permutation.len() != rank {
            return Err(anyhow!("permutation must have the same length as the rank of the layout"));
        }
        let mut permutation = permutation.to_vec();
        permutation.sort();
        for (i, &p) in permutation.iter().enumerate() {
            if p != i {
                return Err(anyhow!("permutation must be a valid permutation"));
            }
        }

        // can't permute a diagonal layout
        if self.is_diagonal() {
            return Err(anyhow!("cannot permute a diagonal layout"));
        }

        // for a sparse tensor, can only permute the sparse axes
        if self.is_sparse() {
            for i in self.rank() - self.n_dense_axes..self.rank() {
                if permutation[i] != i {
                    return Err(anyhow!("cannot permute dense axes of a sparse layout"));
                }
            }
        }

        // permute shape
        let mut new_shape = self.shape.clone();
        for (i, &p) in permutation.iter().enumerate() {
            new_shape[i] = self.shape[p];
        }

        // permute indices
        let mut new_indices = self.indices.clone();
        for (i, index) in new_indices.iter_mut().enumerate() {
            for (ai, &p) in permutation.iter().enumerate() {
                index[ai] = self.indices[i][p];
            }
        }
        Ok(Layout {
            indices: new_indices,
            shape: new_shape,
            kind: self.kind.clone(),
            n_dense_axes: self.n_dense_axes,
        })
    }
    
    // create a new layout by broadcasting a list of layouts
    fn broadcast(layouts: &[Layout]) -> Result<Layout> {
        // the shapes of the layouts must be broadcastable
        let shapes = layouts.iter().map(|x| &x.shape).collect::<Vec<_>>();
        let shape = match broadcast_shapes(&shapes[..]) {
            Some(x) => x,
            None => {
                let shapes_str = shapes.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join(", ");
                return Err(anyhow!("cannot broadcast shapes [{}]", shapes_str));
            }
        };
        let rank = shape.len();
        
        // if the final shape has greater rank than any of the layouts, then add dense axes to those layouts
        // make sure to also update the shape of the layout
        for layout in layouts {
            layout.n_dense_axes += rank - layout.rank();
            layout.shape = shape;
        }

        let indices = layouts[0].indices;
        let n_dense_axes = layouts[0].n_dense_axes;

        // if there are any diagonal layouts then the result is diagonal, all the layouts must be diagonal and have the same number of dense axes
        if layouts.iter().any(|x| x.is_diagonal()) {
            if layouts.iter().any(|x| !x.is_diagonal()) {
                return Err(anyhow!("cannot broadcast diagonal and non-diagonal layouts"));
            }
            if layouts.iter().any(|x| x.n_dense_axes != n_dense_axes) {
                return Err(anyhow!("cannot broadcast diagonal layouts with different numbers of dense axes"));
            } 
            return Ok(Layout {
                indices,
                shape,
                kind: LayoutKind::Diagonal,
                n_dense_axes,
            });
        }
        // if there are any sparse layouts then the result is sparse, and the indicies of all sparse layouts must be identical and have the same number of dense axis. 
        // sparse layouts can be combined with dense layouts.
        if layouts.iter().any(|x| x.is_sparse()) {
            for layout in layouts {
                if layout.is_sparse() {
                    if layout.indices != indices {
                        return Err(anyhow!("cannot broadcast layouts with different sparsity patterns"));
                    }
                    if layout.n_dense_axes != n_dense_axes {
                        return Err(anyhow!("cannot broadcast layouts with different numbers of dense axes"));
                    }
                }
            }
            return Ok(Layout {
                indices,
                shape,
                kind: LayoutKind::Sparse,
                n_dense_axes,
            });
        }
        
        // must be all dense here
        let kind = LayoutKind::Dense;
        Ok(Layout {
            indices,
            shape,
            kind,
            n_dense_axes,
        })
    }
    fn is_dense(&self) -> bool {
        self.kind == LayoutKind::Dense
    }
    fn is_sparse(&self) -> bool {
        self.kind == LayoutKind::Sparse
    }
    fn is_diagonal(&self) -> bool {
        self.kind == LayoutKind::Diagonal
    }
    
    fn into_sparse(&self) -> Self {
        let new_self = self.clone();
        if self.is_dense() {
            new_self.kind = LayoutKind::Sparse;
            new_self.indices = (0..self.shape.iter().product()).map(|x| Self::unravel_index(x, &self.shape)).collect();
        } else if self.is_diagonal() {
            new_self.kind = LayoutKind::Sparse;
            new_self.indices = (0..i64::try_from(self.shape[0]).unwrap()).map(|x| Index::zeros(self.rank()) + x).collect();
        }
        new_self
    }
    
    fn to_sparse(&mut self) {
        if self.is_dense() {
            self.kind = LayoutKind::Sparse;
            self.indices = (0..self.shape.iter().product()).map(|x| Self::unravel_index(x, &self.shape)).collect();
        } else if self.is_diagonal() {
            self.kind = LayoutKind::Sparse;
            self.indices = (0..i64::try_from(self.shape[0]).unwrap()).map(|x| Index::zeros(self.rank()) + x).collect();
        }
    }

    fn new_empty(rank: usize) -> Self {
        Layout {
            indices: vec![],
            shape: Shape::zeros(rank),
            kind: LayoutKind::Dense,
            n_dense_axes: rank,
        }
    }

    fn new_scalar() -> Self {
        Layout {
            indices: vec![],
            shape: Shape::zeros(0),
            kind: LayoutKind::Dense,
            n_dense_axes: 0,
        }
    }

    fn new_dense(shape: Shape) -> Self {
        Layout {
            indices: vec![],
            shape,
            kind: LayoutKind::Dense,
            n_dense_axes: shape.len(),
        }
    }
    
    fn new_diagonal(shape: Shape) -> Self {
        Layout {
            indices: vec![],
            shape,
            kind: LayoutKind::Diagonal,
            n_dense_axes: 0,
        }
    }

    // append another layout to this one along a given axis. the layout is modified in-place
    // can only append if the ranks are the same
    fn append(&mut self, other: &Layout, start: &Index) -> Result<()> {
        if self.rank() != other.rank() {
            return Err(anyhow!("cannot append layouts with different ranks"));
        }
        
        // number of final dense axes must be the same
        if self.n_dense_axes != other.n_dense_axes {
            return Err(anyhow!("cannot append layouts with different numbers of final dense axes"));
        }
        
        // start indices must be zero for final dense axes
        if start.slice(s![self.rank() - self.n_dense_axes..]).iter().any(|&x| x != 0) {
            return Err(anyhow!("start indices must be zero for final dense axes"));
        }
        
        // expand shape to fit the other layout
        let mut new_shape = self.shape.clone();
        let rank = self.rank();
        for i in 0..rank {
            new_shape[i] = max(new_shape[i], other.shape[i] + usize::try_from(start[i]).unwrap());
        }
        
        // if both layouts are dense then we can just update the shape and return
        if self.is_dense() && other.is_dense() {
            // check that the start index is zero for all axes except the first
            if start.iter().skip(1).any(|&x| x != 0) {
                return Err(anyhow!("can only append dense layouts with a start index of zero for all axes except the first"));
            }
            self.shape = new_shape;
            return Ok(());
        }
        
        // if both layouts are diagonal and the start indices are the same then we can just update the shape and return
        if self.is_diagonal() && other.is_diagonal() {
            // check that the start index is on the diagonal
            let first = start[0];
            if start.iter().any(|&x| x != first) {
                return Err(anyhow!("can only append diagonal layouts with a start index on the diagonal"));
            }
            self.shape = new_shape;
            return Ok(());
        }
        
        // if both layouts are sparse then we can just append the indices
        if self.is_sparse() && other.is_sparse() {
            self.indices.extend(other.indices.iter().map(|x| x + start));
            self.shape = new_shape;
            return Ok(());
        }
        
        if self.is_sparse() {
            // if this layout is sparse then convert the other and extend the indices
            let sparse_other = other.into_sparse();
            self.indices.extend(sparse_other.indices().iter().map(|x| x + start));
        } else {
            // if the other layout is sparse then we need to convert to sparse
            let sparse_self = self.into_sparse();
            self.indices = sparse_self.indices;
            self.indices.extend(other.indices.iter().map(|x| x + start));
        }
        self.shape = new_shape;
        Ok(())
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    fn dense(shape: Shape) -> Self {
        Self {
            indices: Vec::new(),
            shape,
            kind: LayoutKind::Dense,
            n_dense_axes: 0,
        }
    }
    fn diagonal(shape: Shape) -> Self {
        Self {
            indices: Vec::new(),
            shape,
            kind: LayoutKind::Diagonal,
            n_dense_axes: 0,
        }
    }
    fn sparse(indices: Vec<Index>, shape: Shape) -> Self {
        Self {
            indices,
            shape,
            kind: LayoutKind::Sparse,
            n_dense_axes: 0,
        }
    }

    fn nnz(&self) -> usize {
        let n_dense: usize = self.shape.slice(s![self.rank() - self.n_dense_axes..]).iter().product();
        if self.is_dense() {
            self.shape.iter().product()
        } else if self.is_diagonal() {
            n_dense * (if self.shape.is_empty() { 0 } else { self.shape[0] })
        } else {
            n_dense * self.indices.len()
        }
    }

    fn indices(&self) -> &Vec<Index> {
        &self.indices
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn kind(&self) -> &LayoutKind {
        &self.kind
    }
}


// RcLayout is a wrapper for Rc<Layout> that implements Hash and PartialEq based on ptr equality
#[derive(Debug)]
struct RcLayout(Rc<Layout>);
impl Hash for RcLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}
impl PartialEq for RcLayout {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Clone for RcLayout {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Eq for RcLayout {}
impl Deref for RcLayout {
    type Target = Layout;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl RcLayout {
    fn new(layout: Layout) -> Self {
        Self(Rc::new(layout))
    }
    fn as_ref(&self) -> &Layout {
        &self.0
    }
}

#[derive(Debug, Clone)]
// F(t, u, u_dot) = G(t, u)
pub struct TensorBlock<'s> {
    name: Option<&'s str>,
    start: Index,
    indices: Vec<char>,
    layout: RcLayout,
    expr_layout: RcLayout,
    expr: Ast<'s>,
}

impl<'s> TensorBlock<'s> {
    pub fn new(name: Option<&'s str>, start: Index, indices: Vec<char>, layout: RcLayout, expr_layout: RcLayout, expr: Ast<'s>) -> Self {
        Self {
            name,
            start,
            indices,
            layout,
            expr_layout,
            expr,
        }
    }
    pub fn new_dense_vector(name: Option<&'s str>, start: i64, shape: usize, expr: Ast<'s>) -> Self {
        let layout = RcLayout::new(Layout::dense(Shape::from_vec(vec![shape])));
        Self {
            name,
            start: Index::from_vec(vec![start]),
            layout: layout.clone(),
            expr_layout: layout,
            expr,
            indices: Vec::new(),
        }
    }
    
    pub fn nnz(&self) -> usize {
        if self.is_diagonal() { self.shape().iter().product() } else { *self.shape().get(0).unwrap_or(&0usize) }
    }

    pub fn shape(&self) -> &Shape {
        &self.layout.shape
    }

    pub fn start(&self) -> &Index {
        &self.start
    }

    pub fn expr(&self) -> &Ast<'s> {
        &self.expr
    }
    
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn is_diagonal(&self) -> bool {
        self.layout.is_diagonal()
    }
    
    pub fn layout(&self) -> &Layout {
        self.layout.as_ref()
    }
}

impl<'s> fmt::Display for TensorBlock<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.rank() > 1 {
            let sep = if self.is_diagonal() { ".." } else { ":" };
            write!(f, "(")?;
            for i in 0..self.rank() {
                write!(f, "{}", self.start[i])?;
                if self.shape()[0] > 1 {
                    write!(f, "{}{}", sep, self.start[i] + i64::try_from(self.shape()[i]).unwrap())?;
                }
                if i < self.rank() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "): ")?;
        }
        write!(f, "{}", self.expr)
    }
}

#[derive(Debug, Clone)]
// F(t, u, u_dot) = G(t, u)
pub struct Tensor<'s> {
    name: &'s str,
    elmts: Vec<TensorBlock<'s>>,
    layout: RcLayout,
    indices: Vec<char>,
}

impl<'s> Tensor<'s> {
    pub fn new_empty(name: &'s str) -> Self {
        Self {
            name,
            elmts: Vec::new(),
            indices: Vec::new(),
            layout: RcLayout::new(Layout::dense(Shape::zeros(0))),
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

    
    pub fn new(name: &'s str, elmts: Vec<TensorBlock<'s>>, indices: Vec<char>) -> Self {
        let rank = elmts.iter().fold(0, |acc, i| max(acc, i.rank()));
        let layout = elmts.iter().fold(Layout::new_empty(rank), |mut acc, i| {
            acc.append(i.layout(), i.start());
            acc
        });
        let layout = RcLayout::new(layout);
        Self {
            name,
            elmts,
            indices,
            layout,
        }
    }


    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.layout.shape()
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn elmts(&self) -> &[TensorBlock] {
        self.elmts.as_ref()
    }
    
    pub fn indices(&self) -> &[char] {
        self.indices.as_ref()
    }

    pub fn layout_ptr(&self) -> &RcLayout {
        &self.layout
    }

    pub fn layout(&self) -> &Layout {
        self.layout.as_ref()
    }
}

impl<'s> fmt::Display for Tensor<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.indices.len() > 0 {
            write!(f, "{}_", self.name).and_then(|_| self.indices.iter().fold(Ok(()), |acc, i| {
                acc.and_then(|_| write!(f, "{}", i))
            }))
        } else {
            write!(f, "{}", self.name)
        }.and_then(|_|  write!(f, " {{\n"))
        .and_then(|_| self.elmts.iter().fold(Ok(()), |acc, e| {
            acc.and_then(|_| write!(f, "  {},\n", e))
        }))
        .and_then(|_| write!(f, "}}"))
    }
}

pub type Shape = Array1<usize>;
pub type Index = Array1<i64>;

struct EnvVar {
    layout: RcLayout,
    is_time_dependent: bool,
    is_state_dependent: bool,
    is_algebraic: bool,
}

impl EnvVar {
    fn is_time_dependent(&self) -> bool {
        self.is_time_dependent
    }

    fn is_state_dependent(&self) -> bool {
        self.is_state_dependent
    }

    fn is_algebraic(&self) -> bool {
        self.is_algebraic
    }

    fn layout(&self) -> &Layout {
        self.layout.as_ref()
    }
}

struct Env<'s> {
    errs: ValidationErrors,
    vars: HashMap<&'s str, EnvVar>,
}

pub fn broadcast_shapes(shapes: &[&Shape]) -> Option<Shape> {
    if shapes.is_empty() {
        return None;
    }
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap();
    let mut shape = Shape::zeros(max_rank);
    for i in (0..max_rank).rev() {
        let (mdim, compatible) = shapes.iter().map(|s| s.get(i).unwrap_or(&1)).fold(
            (1, true),
            |(mdim, _result), dim| {
                let new_mdim = max(mdim, *dim);
                (new_mdim, *dim == 1 || *dim == new_mdim)
            },
        );
        if !compatible {
            return None;
        }
        shape[i] = mdim;
    }
    Some(shape)
}

pub fn can_broadcast_to(to_shape: &Shape, from_shape: &Shape) -> bool {
    let bc_shape = broadcast_shapes(&[to_shape, from_shape]);
    bc_shape.is_some() && bc_shape.unwrap() == *to_shape
}

impl<'s> Env<'s> {
    pub fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert(
            "t",
            EnvVar {
                layout: RcLayout::new(Layout::new_scalar()),
                is_time_dependent: true,
                is_state_dependent: false,
                is_algebraic: true,
            },
        );
        Env {
            errs: ValidationErrors::new(),
            vars,
        }
    }
    pub fn is_tensor_time_dependent(&self, tensor: &Tensor) -> bool {
        tensor.elmts.iter().any(|block| {
            block
                .expr
                .get_dependents()
                .iter()
                .any(|&dep| dep == "t" || self.vars[dep].is_time_dependent())
        })
    }
    pub fn is_tensor_state_dependent(&self, tensor: &Tensor) -> bool {
        tensor.elmts.iter().any(|block| {
            block
                .expr
                .get_dependents()
                .iter()
                .any(|&dep| dep == "u" || self.vars[dep].is_state_dependent())
        })
    }

    pub fn push_var(&mut self, var: &Tensor<'s>) {
        self.vars.insert(
            var.name,
            EnvVar {
                layout: var.layout_ptr().clone(),
                is_algebraic: true,
                is_time_dependent: self.is_tensor_time_dependent(var),
                is_state_dependent: self.is_tensor_state_dependent(var),
            },
        );
    }


    fn get(&self, name: &str) -> Option<&EnvVar> {
        self.vars.get(name)
    }
    fn get_layout_binary_op(
        &mut self,
        left: &Ast<'s>,
        right: &Ast<'s>,
        indices: &Vec<char>,
    ) -> Option<Layout> {
        let mut left_layout = self.get_layout(left, indices)?;
        let mut right_layout = self.get_layout(right, indices)?;
        match Layout::broadcast(&[left_layout, right_layout]) {
            Ok(layout) => Some(layout),
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}", e),
                    left.span,
                ));
                None
            }
        }
    }
    fn get_layout_dot(
        &mut self,
        call: &Call<'s>,
        ast: &Ast<'s>,
        indices: &Vec<char>,
    ) -> Option<Layout> {
        if call.args.len() != 1 {
            self.errs.push(ValidationError::new(
                format!(
                    "time derivative dot call expects 1 argument, got {}",
                    call.args.len()
                ),
                ast.span,
            ));
            return None;
        }
        let arg = &call.args[0];
        let arg_layout = self.get_layout(arg, indices)?;
        if arg_layout.rank() != 1 {
            self.errs.push(ValidationError::new(
                format!(
                    "time derivative dot call expects 1D argument, got {}D",
                    arg_layout.rank()
                ),
                ast.span,
            ));
            return None;
        }
        let depends_on = arg.get_dependents();
        // for each state variable, set is_algebraic to false
        for dep in depends_on {
            if let Some(var) = self.vars.get_mut(dep) {
                var.is_algebraic = false;
            }
        }
        return Some(arg_layout);
    }

    fn get_layout_name(
        &mut self,
        name: &str,
        ast: &Ast,
        rhs_indices: &Vec<char>,
        lhs_indices: &Vec<char>,
    ) -> Option<Layout> {
        let var = self.get(name);
        if var.is_none() {
            self.errs.push(ValidationError::new(
                format!("cannot find variable {}", name),
                ast.span,
            ));
            return None;
        }
        let var = var.unwrap();
        let layout = var.layout();

        if rhs_indices.len() < layout.rank() {
            self.errs.push(ValidationError::new(
                format!(
                    "cannot index variable {} with {} indices. Expected at least {} indices",
                    name,
                    rhs_indices.len(),
                    layout.rank()
                ),
                ast.span,
            ));
            return None;
        }
        let permutation = vec![0; rhs_indices.len()];
        for i in 0..rhs_indices.len() {
            permutation[i] = match lhs_indices.iter().position(|&x| x == rhs_indices[i]) {
                Some(pos) => pos,
                None => {
                    self.errs.push(ValidationError::new(
                        format!("cannot find index {} in lhs indices {:?} ", rhs_indices[i], lhs_indices),
                        ast.span,
                    ));
                    return None;
                }
            }
        }
        let layout_permuted = match layout.permute(permutation.as_slice()) {
            Ok(layout) => layout,
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}", e),
                    ast.span,
                ));
                return None;
            }
        };
                
        Some(layout_permuted)
    }

    

    fn get_layout_call(&mut self, call: &Call<'s>, ast: &Ast, indices: &Vec<char>) -> Option<Layout> {
        let mut layouts = call
            .args
            .iter()
            .map(|c| self.get_layout(c, indices))
            .collect::<Option<Vec<Layout>>>()?;
        match Layout::broadcast(layouts.as_slice()) {
            Ok(layout) => Some(layout),
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}", e),
                    ast.span,
                ));
                None
            }
        }    
    }

    pub fn get_layout(&mut self, ast: &Ast<'s>, indices: &Vec<char>) -> Option<Layout> {
        match &ast.kind {
            AstKind::Assignment(a) => self.get_layout(a.expr.as_ref(), indices),
            AstKind::Binop(binop) => {
                self.get_layout_binary_op(binop.left.as_ref(), binop.right.as_ref(), indices)
            }
            AstKind::Monop(monop) => self.get_layout(monop.child.as_ref(), indices),
            AstKind::Call(call) => match call.fn_name {
                "dot" => self.get_layout_dot(&call, ast, indices),
                _ => self.get_layout_call(&call, ast, indices),
            },
            AstKind::CallArg(arg) => self.get_layout(arg.expression.as_ref(), indices),
            AstKind::Index(i) => {
                self.get_layout_binary_op(i.left.as_ref(), i.right.as_ref(), indices)
            }
            AstKind::Slice(s) => {
                self.get_layout_binary_op(s.lower.as_ref(), s.upper.as_ref(), indices)
            }
            AstKind::Number(_) => Some(Layout::new_scalar()),
            AstKind::Integer(_) => Some(Layout::new_scalar()),
            AstKind::Domain(d) => Some(Layout::new_dense(Shape::zeros(1) + d.dim)),
            AstKind::IndexedName(name) => {
                self.get_layout_name(name.name, ast, &name.indices, indices)
            }
            AstKind::Name(name) => self.get_layout_name(name, ast, &vec![], indices),
            _ => panic!("unrecognised ast node {:#?}", ast.kind),
        }
    }
    

    // returns a tuple of (expr_layout, elmt_layout) giving the layouts of the expression and the tensor element.)
    fn get_layout_tensor_elmt(&mut self, elmt: &TensorElmt<'s>, indices: &Vec<char>) -> Option<(Layout, Layout)> {
        let expr_indices = elmt.expr.get_indices();
        // get any indices from the expression that do not appear in 'indices' and add them to 'indices' to a new vector
        let mut new_indices = indices.clone();
        for i in expr_indices {
            if !indices.contains(&i) {
                new_indices.push(i);
            }
        }
        
        // TODO: for now we will only support one additional index
        if new_indices.len() > indices.len() + 1 {
            self.errs.push(ValidationError::new(
                format!(
                    "cannot index tensor element with more than one additional index. Found {} indices",
                    new_indices.len() - indices.len()
                ),
                elmt.expr.span,
            ));
            return None;
        }

        let mut expr_layout = self.get_layout(elmt.expr.as_ref(), &new_indices)?;
        
        // if we have an additional index then we contract the last dimension of the expression layout to get the final layout
        if new_indices.len() > indices.len() {
            expr_layout = match expr_layout.contract_last_axis() {
                Ok(layout) => layout,
                Err(e) => {
                    self.errs.push(ValidationError::new(
                        format!("{}", e),
                        elmt.expr.span,
                    ));
                    return None;
                }
            }
        };
        
        // calculate the shape of the tensor element. 
        let elmt_layout = if elmt.indices.is_none() {

            // If there are no indices then the rank of the expression must be 0 or 1 (i.e a scalar
            // or a vector) and if 1 then the length of the vector is the same as the expression
            if expr_layout.rank() == 0 {
                Layout::new_scalar()
            } else if expr_layout.rank() == 1 {
                Layout::new_dense(expr_layout.shape().clone())
            } else {
                self.errs.push(ValidationError::new(
                    format!(
                        "tensor element without indices must be a scalar or vector, but expression has rank {}",
                        expr_layout.rank()
                    ),
                    elmt.expr.span,
                ));
                return None;
            }
        } else {
            // If there are indicies then the rank is determined by the number of indices, and the
            // shape is determined by the ranges of the indices
            // TODO: this is quite large, perhaps move to another function
            

            // make sure the number of indices matches the number of dimensions
            let elmt_indices = elmt.indices.as_ref().unwrap();
            let given_indices_ast = &elmt_indices.kind.as_vector().unwrap().data;
            let given_indices: Vec<&Indice> = given_indices_ast.iter().map(|i| i.kind.as_indice().unwrap()).collect();
            if given_indices.len() != indices.len() {
                self.errs.push(ValidationError::new(
                    format!(
                        "number of dimensions of tensor element ({}) does not match number of dimensions of tensor ({})",
                        given_indices.len(), indices.len()
                    ),
                    elmt_indices.span,
                ));
                return None;
            }
            
            let mut exp_expr_shape = Shape::ones(indices.len());
            
            // we will use the expression shape as defaults if the range is not explicitly given
            exp_expr_shape.slice_mut(s![..expr_layout.rank()]).assign(&expr_layout.shape());
            
            // calculate the shape of the tensor element from the given indices and expression shape
            let all_range_indices = given_indices.iter().all(|i| i.sep == Some(".."));
            let mut old_dim = None;
            for (i, indice) in given_indices.iter().enumerate() {
                let first = indice.first.kind.as_integer().unwrap();
                
                // make sure the use of the range separator is valid
                if !all_range_indices && matches!(indice.sep, Some("..")) {
                    self.errs.push(ValidationError::new(
                        format!("can only use range separator if all indices are ranges"),
                        given_indices_ast[i].span,
                    ));
                }
                let dim = if let Some(_) = indice.sep {
                    if let Some(second) = &indice.last {
                        let second = second.kind.as_integer().unwrap();
                        if second < first {
                            self.errs.push(ValidationError::new(
                                format!("range end must be greater than range start"),
                                given_indices_ast[i].span,
                            ));
                            return None;
                        }
                        usize::try_from(second - first).unwrap()
                    } else {
                        exp_expr_shape[i]
                    }
                } else {
                    1usize
                };
                
                // make sure the dimension of the range is consistent
                if all_range_indices && old_dim.is_some() && dim != old_dim.unwrap() {
                    self.errs.push(ValidationError::new(
                        format!("range indices must have the same dimension"),
                        given_indices_ast[i].span,
                    ));
                    return None;
                }
                old_dim = Some(dim);
                exp_expr_shape[i] = dim;
            }
                
            // tensor elmt layout is:
            // 1. dense if the expression is dense and no indices are ranges
            // 2. diagonal if the expression is dense and all indices are ranges, or the expression is diagonal and no indices are ranges
            // 3. sparse if the expression is sparse and no indices are blocks
            let elmt_layout = match expr_layout.kind() {
                LayoutKind::Dense => {
                    if all_range_indices {
                        Layout::new_diagonal(exp_expr_shape)
                    } else {
                        Layout::new_dense(exp_expr_shape)
                    }
                },
                LayoutKind::Sparse => {
                    if all_range_indices {
                        self.errs.push(ValidationError::new(
                            format!("cannot use range indices with sparse expression"),
                            elmt.expr.span,
                        ));
                        return None;
                    } else {
                        expr_layout    
                    }
                },
                LayoutKind::Diagonal => {
                    if all_range_indices {
                        self.errs.push(ValidationError::new(
                            format!("cannot use range indices with diagonal expression"),
                            elmt.expr.span,
                        ));
                        return None;
                    } else {
                        Layout::new_diagonal(exp_expr_shape)
                    }
                },
            };
            elmt_layout
        };

        // now broadcast the expr_layout and the elmt_layout to get the final layout
        let layouts = vec![expr_layout, elmt_layout];
        let elmt_layout = match Layout::broadcast(&[expr_layout, elmt_layout]) {
            Ok(layout) => layout,
            Err(e) => {
                self.errs.push(ValidationError::new(
                    format!("{}", e),
                    elmt.expr.span,
                ));
                return None;
            }
        };
        
        Some((expr_layout, elmt_layout))
    }
}


// there are three different layouts:
// 1. the data layout is a mapping from tensors to the index of the first element in the data array. Each tensor in the data layout is a contiguous array of nnz elements
// 2. the layout layout is a mapping from Layout to the index of the first element in the data array. Only sparse layouts are stored, and each sparse layout is a contiguous array of nnz*rank elements
// 3. the contraction layout is a mapping from layout from-to pairs to the index of the first element in the data array. Only contraction pairs are stored, and each contraction pair is a contiguous array of nnz elements where nnz is the number of non-zero elements in the from layout
#[derive(Debug)]
pub struct DataLayout {
    data_index_map: HashMap<String, usize>,
    data_length: usize,
    layout_index_map: HashMap<Layout, usize>,
    layout_length: usize,
    contraction_index_map: HashMap<(String, String), usize>,
}

impl DataLayout {
    pub fn new(model: &DiscreteModel) -> Self {
        
        let mut tensor_map = HashMap::new();
        let mut data_index_map = HashMap::new();
        let mut layout_index_map = HashMap::new();
        let mut i = 0usize;
        let mut ilayout = 0usize;

        data_index_map.insert("inputs".to_string(), i);
        i += model.inputs.nnz();

        for dfn in model.time_indep_defns.iter().chain(model.time_dep_defns.iter()) {
            data_index_map.insert(dfn.name.to_string(), i);
            tensor_map.insert(dfn.name.to_string(), &dfn);
            i += dfn.nnz();
        }

        data_index_map.insert("states".to_string(), i);
        i += model.state.nnz();

        for dfn in model.state_dep_defns.iter() {
            data_index_map.insert(dfn.name.to_string(), i);
            tensor_map.insert(dfn.name.to_string(), &dfn);
            i += dfn.nnz();
        }

        data_index_map.insert(model.lhs.name.to_string(), i);
        i += model.lhs.nnz();

        data_index_map.insert(model.rhs.name.to_string(), i);
        i += model.rhs.nnz();

        let data_length = i;

        Self { data_index_map, data_length }
    }
    
    pub fn data_length(&self) -> usize {
        self.data_length
    }

    // get the index of the data array for the given tensor name
    pub fn get_data_index(&self, name: &str) -> Option<usize> {
        self.data_index_map.get(name).map(|i| *i)
    }

}


#[derive(Debug)]
// F(t, u, u_dot) = G(t, u)
pub struct DiscreteModel<'s> {
    pub name: &'s str,
    pub lhs: Tensor<'s>,
    pub rhs: Tensor<'s>,
    pub out: Tensor<'s>,
    pub time_indep_defns: Vec<Tensor<'s>>,
    pub time_dep_defns: Vec<Tensor<'s>>,
    pub state_dep_defns: Vec<Tensor<'s>>,
    pub inputs: Vec<Tensor<'s>>,
    pub state: Tensor<'s>,
}

impl<'s, 'a> fmt::Display for DiscreteModel<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inputs.iter().fold(Ok(()), |acc, input| acc.and_then(|_| write!(f, "{}\n", input)))
        .and_then(|_| self.time_indep_defns.iter().fold(Ok(()), |acc, defn| acc.and_then(|_| write!(f, "{}\n", defn))))
        .and_then(|_| self.time_dep_defns.iter().fold(Ok(()), |acc, defn| acc.and_then(|_| write!(f, "{}\n", defn))))
        .and_then(|_| write!(f, "{}\n", self.state))
        .and_then(|_| self.state_dep_defns.iter().fold(Ok(()), |acc, defn| acc.and_then(|_| write!(f, "{}\n", defn))))
        .and_then(|_| write!(f, "{}\n", self.lhs))
        .and_then(|_| write!(f, "{}\n", self.rhs))
        .and_then(|_| write!(f, "{}\n", self.out))
    }
}

impl<'s> DiscreteModel<'s> {
    pub fn new(name: &'s str) -> Self {
        Self {
            name,
            lhs: Tensor::new_empty("F"),
            rhs: Tensor::new_empty("G"),
            out: Tensor::new_empty("out"),
            time_indep_defns: Vec::new(),
            time_dep_defns: Vec::new(),
            state_dep_defns: Vec::new(),
            inputs: Vec::new(),
            state: Tensor::new_empty("u"),
        }
    }

    // define data layout for storage in a single array of arrays
    // need 1 array for each dense tensor
    // need 2 arrays (data + layout) for each sparse tensor
    // don't need anything for out
    pub fn create_data_layout<'t>(&'t self) -> DataLayout {
        DataLayout::new(self)
    }

    // residual = F(t, u, u_dot) - G(t, u)
    // return a tensor equal to the residual
    pub fn residual(&self) -> Tensor<'s> {
        let mut residual = self.lhs.clone();
        residual.name = "residual";
        let lhs = Ast {
            kind: AstKind::new_name("F"),
            span: None,
        };
        let rhs = Ast {
            kind: AstKind::new_name("G"),
            span: None,
        };
        residual.elmts = vec![
            TensorBlock {
                name: None,
                expr: Ast {
                    kind: AstKind::new_binop('-', lhs, rhs),
                    span: None,
                },
                start: Index::from_vec(vec![0]),
                indices: vec!['i'],
                layout: self.lhs.layout_ptr().clone(),
                expr_layout: self.lhs.layout_ptr().clone(),
            };
            residual.elmts.len()
        ];
        residual
    }

    
    fn build_domain(domain_ast: &Ast<'s>, env: &mut Env<'s>) -> (f64, f64) {
        let domain = domain_ast.kind.as_domain().unwrap();
        match &domain.range.kind {
            AstKind::Range(r) => (r.lower, r.upper),
            AstKind::Name(name) => match *name {
                "R" => (-f64::INFINITY, f64::INFINITY),
                _ => {
                    env.errs.push(ValidationError::new(
                        format!("Unknown domain {}", name),
                        domain_ast.span,
                    ));
                    (-f64::INFINITY, f64::INFINITY)
                }
            },
            _ => {
                env.errs.push(ValidationError::new(
                    format!("Unknown domain, should be a range or a name"),
                    domain_ast.span,
                ));
                (-f64::INFINITY, f64::INFINITY)
            }
        }
    }

    fn build_array(array: &ast::Tensor<'s>, env: &mut Env<'s>) -> Option<Tensor<'s>> {
        let rank = array.indices.len();
        let mut elmts = Vec::new();
        let mut start = Index::zeros(rank);
        let nerrs = env.errs.len();
        if rank == 0 && array.elmts.len() > 1 {
            env.errs.push(ValidationError::new(
                format!("cannot have more than one element in a scalar"),
                array.elmts[1].span,
            ));
        }
        let mut tensor_layout = Layout::new_empty(rank);
        for a in &array.elmts {
            match &a.kind {
                AstKind::TensorElmt(te) => {
                    if let Some((expr_layout, elmt_layout)) = env.get_layout_tensor_elmt(&te, &array.indices) {
                        if rank == 0 && elmt_layout.rank() == 1 {
                            if elmt_layout.shape()[0] > 1 {
                                env.errs.push(ValidationError::new(
                                    format!("cannot assign an expression with rank > 1 to a scalar, rhs has shape {}", elmt_layout.shape()),
                                    a.span,
                                ));
                            }
                        }
                        let (name, expr) = if let AstKind::Assignment(a) = &te.expr.kind {
                            (Some(a.name), a.expr.clone())
                        } else {
                            (None, te.expr.clone())
                        };
                        elmts.push(TensorBlock::new(name, start.clone(), array.indices, RcLayout::new(elmt_layout), RcLayout::new(expr_layout), *te.expr.clone()));
                        tensor_layout.append(&elmt_layout, &start);
                        start += &elmt_layout.shape().mapv(|x| i64::try_from(x).unwrap());
                    }
                },
                _ => unreachable!("unexpected expression in tensor definition"),
            }
        }
        let tensor = Tensor::new(array.name, elmts, array.indices);
        if nerrs == env.errs.len() {
            env.push_var(&tensor);
        }
        Some(tensor)
    }


    fn check_match(tensor1: &Tensor, tensor2: &Tensor, span: Option<StringSpan>, env: &mut Env) {
        // check shapes
        if tensor1.shape() != tensor2.shape() {
            env.errs.push(ValidationError::new(
                format!(
                    "{} and {} must have the same shape, but {} has shape {} and {} has shape {}",
                    tensor1.name,
                    tensor2.name,
                    tensor1.name,
                    tensor2.name,
                    tensor1.shape(),
                    tensor2.shape()
                ),
                span,
            ));
        }
    }

    pub fn build(name: &'s str, model: &ast::DsModel) -> Result<Self, ValidationErrors> {
        let mut env = Env::new();
        let mut ret = Self::new(name);
        let mut read_state = false;
        let mut read_out = false;
        let span_f = None;
        let span_g = None;
        for (i, tensor_ast) in model.tensors.iter().enumerate() {
            match tensor_ast.kind.as_array() {
                None => env.errs.push(ValidationError::new(
                    "not an array".to_string(),
                    tensor_ast.span,
                )),
                Some(tensor) => {
                    let span = tensor_ast.span;
                    match tensor.name {
                        "u" => {
                            read_state = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.state = built;
                            }
                            if ret.state.rank() > 1 {
                                env.errs.push(ValidationError::new(
                                    "u must be a scalar or 1D vector".to_string(),
                                    span,
                                ));
                            }
                        }
                        "F" => {
                            span_f = Some(span);
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.lhs = built;
                            }
                        }
                        "G" => {
                            span_g = Some(span);
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                ret.rhs = built;
                            }
                        }
                        "out" => {
                            read_out = true;
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                if built.shape.len() > 1 {
                                    env.errs.push(ValidationError::new(
                                        format!("output shape must be a scalar or 1D vector"),
                                        tensor_ast.span,
                                    ));
                                }
                                print!("built out: {:#?}", built);
                                ret.out = built;
                            }
                        }
                        _name => {
                            if let Some(built) = Self::build_array(tensor, &mut env) {
                                if let Some(env_entry) = env.get(built.name) {
                                    let dependent_on_state = env_entry.is_state_dependent();
                                    let dependent_on_time = env_entry.is_time_dependent();
                                    if !dependent_on_time {
                                        ret.time_indep_defns.push(built);
                                    } else if dependent_on_time && !dependent_on_state {
                                        ret.time_dep_defns.push(built);
                                    } else {
                                        ret.state_dep_defns.push(built);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // set is_algebraic for every state based on env
        for s in &mut ret.states.elmts {
            let env_entry = env.get(s.name).unwrap();
            s.is_algebraic = env_entry.is_algebraic();
        }
        // check that we've read all the required arrays

        let span_all = if ast.is_empty() && ast.first().unwrap().span.is_some() {
            None
        } else {
            Some(StringSpan {
                pos_start: ast.first().unwrap().span.unwrap().pos_start,
                pos_end: ast.last().unwrap().span.unwrap().pos_start,
            })
        };
        if !read_state {
            env.errs.push(ValidationError::new(
                "missing 'u' array".to_string(),
                span_all,
            ));
        }
        if span_f.is_none() {
            env.errs.push(ValidationError::new(
                "missing 'F' array".to_string(),
                span_all,
            ));
        }
        if span_g.is_none() {
            env.errs.push(ValidationError::new(
                "missing 'G' array".to_string(),
                span_all,
            ));
        }
        if !read_out {
            env.errs.push(ValidationError::new(
                "missing 'out' array".to_string(),
                span_all,
            ));
        }
        if let Some(span) = span_f {
            Self::check_match(&ret.lhs, &ret.state, span, &mut env);
        }
        if let Some(span) = span_g {
            Self::check_match(&ret.rhs, &ret.state, span, &mut env);
        }

        if env.errs.is_empty() {
            Ok(ret)
        } else {
            Err(env.errs)
        }
    }
    
    fn state_to_elmt(state_cell: &Rc<RefCell<Variable<'s>>>) -> (TensorBlock<'s>, TensorBlock<'s>) {
        let state = state_cell.borrow();
        let ast_eqn = if let Some(eqn) = &state.equation {
            eqn.clone()
        } else {
            panic!("state var should have an equation")
        };
        let (f_astkind, g_astkind) = match ast_eqn.kind {
            AstKind::RateEquation(eqn) => (
                AstKind::new_dot(Ast {
                    kind: AstKind::new_name(state.name),
                    span: ast_eqn.span,
                }),
                eqn.rhs.kind,
            ),
            AstKind::Equation(eqn) => (
                AstKind::new_num(0.0),
                AstKind::new_binop('-', *eqn.rhs, *eqn.lhs),
            ),
            _ => panic!("equation for state var should be rate eqn or standard eqn"),
        };
        (
            TensorBlock::new_dense_vector(0, state.dim, Ast { kind: f_astkind, span: ast_eqn.span }),
            TensorBlock::new_dense_vector(0, state.dim, Ast { kind: g_astkind, span: ast_eqn.span }),
        )
    }
    fn state_to_u0(state_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let state = state_cell.borrow();
        let init = if state.has_initial_condition() {
            state.init_conditions[0].equation.clone()
        } else {
            Ast { kind: AstKind::new_num(0.0), span: None }
        };
        TensorBlock::new_dense_vector(0, state.dim, init)
    }
    fn dfn_to_array(defn_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let defn = defn_cell.borrow();
        let tsr_blk = TensorBlock::new_dense_vector(0, defn.dim, defn.expression.as_ref().unwrap().clone());
        Tensor::new(defn.name, vec![tsr_blk], vec!['i'])
    }

    fn state_to_input(input_cell: &Rc<RefCell<Variable<'s>>>) -> Tensor<'s> {
        let input = input_cell.borrow();
        assert!(input.is_independent());
        assert!(!input.is_time_dependent());
        Tensor::new(input.name, vec![TensorBlock::new_dense_vector(0, input.dim, Ast { kind: AstKind::new_name(input.name), span: None })], vec!['i'])
    }
    fn output_to_elmt(output_cell: &Rc<RefCell<Variable<'s>>>) -> TensorBlock<'s> {
        let output = output_cell.borrow();
        let expr = Ast {
            kind: AstKind::new_name(output.name),
            span: if output.is_definition() {
                output.expression.as_ref().unwrap().span
            } else if output.has_equation() {
                output.equation.as_ref().unwrap().span
            } else {
                None
            },
        };
        TensorBlock::new_dense_vector(0, output.dim, expr)
    }
    pub fn from(model: &ModelInfo<'s>) -> DiscreteModel<'s> {
        let (time_varying_unknowns, const_unknowns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = model
            .unknowns
            .iter()
            .cloned()
            .partition(|var| var.borrow().is_time_dependent());

        let states: Vec<Rc<RefCell<Variable>>> = time_varying_unknowns
            .iter()
            .filter(|v| v.borrow().is_state())
            .cloned()
            .collect();

        let (state_dep_defns, state_indep_defns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = model
            .definitions
            .iter()
            .cloned()
            .partition(|v| v.borrow().is_dependent_on_state());

        let (time_dep_defns, const_defns): (
            Vec<Rc<RefCell<Variable>>>,
            Vec<Rc<RefCell<Variable>>>,
        ) = state_indep_defns
            .iter()
            .cloned()
            .partition(|v| v.borrow().is_time_dependent());

        let mut out_array_elmts: Vec<TensorBlock> =
            chain(time_varying_unknowns.iter(), model.definitions.iter())
                .map(DiscreteModel::output_to_elmt)
                .collect();
        let mut curr_index: usize = 0;
        for elmt in out_array_elmts.iter_mut() {
            elmt.start[0] = i64::try_from(curr_index).unwrap();
            curr_index = curr_index + elmt.layout().shape()[0];
        }
        let out_array = Tensor::new(
            "out",
            out_array_elmts,
            vec!['i'],
        );

        let mut f_elmts: Vec<TensorBlock> = Vec::new();
        let mut g_elmts: Vec<TensorBlock> = Vec::new();
        let mut curr_index = 0;
        let mut init_states: Vec<TensorBlock> = Vec::new();
        for state in states.iter() {
            let mut elmt = DiscreteModel::state_to_elmt(state);
            elmt.0.start[0] = i64::try_from(curr_index).unwrap();
            elmt.1.start[0] = i64::try_from(curr_index).unwrap();
            let mut init_state = DiscreteModel::state_to_u0(state);
            init_state.start[0] = i64::try_from(curr_index).unwrap();
            curr_index = curr_index + elmt.0.layout().shape[0];
            f_elmts.push(elmt.0);
            g_elmts.push(elmt.1);
            init_states.push(init_state);
        }
        let state = Tensor::new("u", init_states, vec!['i']);

        let mut inputs: Vec<Tensor> = Vec::new();
        for input in const_unknowns.iter() {
            let mut inp = DiscreteModel::state_to_input(input);
            inputs.push(inp);
        }

        let state_dep_defns = state_dep_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let time_dep_defns = time_dep_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let time_indep_defns = const_defns
            .iter()
            .map(DiscreteModel::dfn_to_array)
            .collect();
        let lhs_shape = &f_elmts
            .last()
            .unwrap()
            .start
            .mapv(|x| usize::try_from(x).unwrap())
            + f_elmts.last().unwrap().layout().shape();
        let lhs =  Tensor::new("F", f_elmts, vec!['i']);
        let rhs_shape = &g_elmts
            .last()
            .unwrap()
            .start
            .mapv(|x| usize::try_from(x).unwrap())
            + g_elmts.last().unwrap().layout().shape();
        let rhs = Tensor::new("G", g_elmts, vec!['i']);
        let name = model.name;
        DiscreteModel {
            name,
            lhs,
            rhs,
            inputs,
            state,
            out: out_array,
            time_indep_defns,
            time_dep_defns,
            state_dep_defns,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::ModelInfo, discretise::DiscreteModel, ds_parser, ms_parser::parse_string,
    };

    #[test]
    fn test_circuit_model() {
        let text = "
        model resistor( i(t), v(t), r -> NonNegative) {
            v = i * r
        }
        model circuit(i(t)) {
            let inputVoltage = sin(t) 
            use resistor(v = inputVoltage)
            let doubleI = 2 * i
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("circuit", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.time_indep_defns.len(), 0);
        assert_eq!(discrete.time_dep_defns.len(), 1);
        assert_eq!(discrete.time_dep_defns[0].name, "inputVoltage");
        assert_eq!(discrete.state_dep_defns.len(), 1);
        assert_eq!(discrete.state_dep_defns[0].name, "doubleI");
        assert_eq!(discrete.lhs.name, "F");
        assert_eq!(discrete.rhs.name, "G");
        assert_eq!(discrete.states.shape()[0], 1);
        assert_eq!(discrete.states.elmts().len(), 1);
        assert_eq!(discrete.out.elmts.len(), 3);
        println!("{}", discrete);
    }
    #[test]
    fn rate_equation() {
        let text = "
        model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), t, z(t) ) { 
            dot(y) = r * y * (1 - y / k)
            y(0) = 1.0
            z = 2 * y
        }
        ";
        let models = parse_string(text).unwrap();
        let model_info = ModelInfo::build("logistic_growth", &models).unwrap();
        assert_eq!(model_info.errors.len(), 0);
        let discrete = DiscreteModel::from(&model_info);
        assert_eq!(discrete.out.elmts[0].expr.to_string(), "y");
        assert_eq!(discrete.out.elmts[1].expr.to_string(), "t");
        assert_eq!(discrete.out.elmts[2].expr.to_string(), "z");
        println!("{}", discrete);
    }

    #[test]
    fn discrete_logistic_model() {
        const TEXT: &str = "
            in_i {
                r -> [0, inf],
                k -> [0, inf],
            }
            u_i {
                y -> R = 1,
                z -> R,
            }
            F_i {
                dot(y),
                0,
            }
            G_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                y,
                t,
                z,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("logistic_growth", &arrays) {
            Ok(model) => {
                let model_str: String = format!("{}", model).chars().filter(|c| !c.is_whitespace()).collect();
                let text_str: String = TEXT.chars().filter(|c| !c.is_whitespace()).collect();
                assert_eq!(model_str, text_str);
                println!("{}", model);
            }
            Err(e) => {
                panic!("{}", e.as_error_message(TEXT));
            }
        };
    }

    #[test]
    fn discrete_logistic_model_single_state() {
        const TEXT: &str = "
            in {
                r -> [0, inf],
            }
            u {
                y -> R = 1,
            }
            F {
                dot(y),
            }
            G {
                (r * y) * (1 - y),
            }
            out {
                y,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("logistic_growth", &arrays) {
            Ok(model) => {
                let model_str: String = format!("{}", model).chars().filter(|c| !c.is_whitespace()).collect();
                let text_str: String = TEXT.chars().filter(|c| !c.is_whitespace()).collect();
                assert_eq!(model_str, text_str);
                println!("{}", model);
            }
            Err(e) => {
                panic!("{}", e.as_error_message(TEXT));
            }
        };
    }

    #[test]
    fn logistic_model_with_matrix() {
        const TEXT: &str = "
            in_i {
                r -> [0, inf],
                k -> [0, inf],
            }
            sm_ij {
                (0..2, 0..2): 1,
            }
            I_ij {
                (0:2, 0:2): sm_ij,
                (2, 2): 1,
                (3, 3): 1,
            }
            u_i {
                y -> R**2 = 1,
                z -> R**2,
            }
            rhs_i {
                (r * y_i) * (1 - (y_i / k)),
                (2 * y_i) - z_i,
            }
            F_i {
                dot(y_i),
                0,
                0,
            }
            G_i {
                sum(j, I_ij * rhs_i),
            }
            out_i {
                y_i,
                t,
                z_i,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("logistic_growth", &arrays) {
            Ok(model) => {
                let model_str: String = format!("{}", model).chars().filter(|c| !c.is_whitespace()).collect();
                let text_str: String = TEXT.chars().filter(|c| !c.is_whitespace()).collect();
                assert_eq!(model_str, text_str);
            }
            Err(e) => {
                panic!("{}", e.as_error_message(TEXT));
            }
        };
    }
 
    
    #[test]
    fn param_error() {
        const TEXT: &str = "
            in_i {
                2 * 1
            }
            u_i {
                y -> R,
            }
            F_i {
                z -> R = 1,
            }
            G {
                y * (1 - (y / k)),
                2 * y
            }
            out_i {
                y,
            }
        ";
        let arrays: Vec<_> = ds_parser::parse_string(TEXT).unwrap();
        match DiscreteModel::build("test", &arrays) {
            Ok(model) => {
                panic!("Should have failed: {}", model)
            }
            Err(e) => {
                assert!(e.has_error_contains("expected parameter in input"));
                assert!(e.has_error_contains("cannot have parameters in tensor"));
                assert!(e.has_error_contains("cannot have more than one element in a scalar"));
                assert!(e.has_error_contains("cannot find variable k"));
                assert!(e.has_error_contains("F and u must have the same shape"));
                assert!(e.has_error_contains("G and u must have the same shape"));
            }
        };
    }
}
