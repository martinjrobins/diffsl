use anyhow::{anyhow, Result};
use ndarray::s;
use std::{
    cmp::min,
    convert::AsRef,
    fmt,
    hash::{Hash, Hasher},
    iter::zip,
    mem,
    ops::Deref,
    sync::Arc,
};

use crate::{ast::AstKind, discretise::Env};

use super::{broadcast_shapes, shape::Shape, Index, TensorBlock};

#[derive(Debug, Clone, PartialEq)]
pub enum LayoutKind {
    Dense,
    Diagonal,
    Sparse,
}

#[derive(Debug, Clone, Copy)]
pub enum TensorType {
    State,
    StateDot,
    Input,
    Other,
}

pub type NonZero = (Index, usize);

/// a sparsity pattern for a multidimensional tensor. A tensor can be sparse, diagonal or dense, as given by the kind field.
/// A tensor can also have n_dense_axes axes which are dense, these are the last n_dense_axes axes of the tensor. So for example,
/// you could have a 2D sparse tensor with 1 dense axis, combining to give a tensor of rank 3 (i.e. the shape is length 3).
/// indices are kept in row major order, so the last index is the fastest changing index.
///
/// For sparse layouts, the indices field contains the list of non-zero indices, excluding the dense axes.
/// For diagonal layouts, the indices field is empty, as the non-zero indices are implicit.
/// For dense layouts, the indices field is empty, as all indices are non-zero.
///
/// Each layout contains a vector of state and input dependencies, which are pairs of (index, state/input_j), indicating that the non-zero at index depends on state/input j.
#[derive(Debug, Clone)]
pub struct Layout {
    indices: Vec<Index>,
    state_deps: Vec<NonZero>,
    input_deps: Vec<NonZero>,
    shape: Shape,
    kind: LayoutKind,
    n_dense_axes: usize,
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            return Ok(());
        }
        write!(f, " (")?;
        for i in 0..self.rank() {
            let type_char = if self.is_diagonal() && i < self.rank() - self.n_dense_axes {
                Some('i')
            } else if self.is_sparse() && i < self.rank() - self.n_dense_axes {
                Some('s')
            } else {
                None
            };
            if let Some(type_char) = type_char {
                write!(f, "{}{}", self.shape()[i], type_char)?;
            } else {
                write!(f, "{}", self.shape()[i])?;
            }
            if i < self.rank() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

impl Layout {
    /// Merge two dependency lists, removing duplicates.
    fn merge_deps(mut a: Vec<(Index, usize)>, mut b: Vec<(Index, usize)>) -> Vec<(Index, usize)> {
        a.append(&mut b);
        a.sort_unstable_by(|x, y| Self::cmp_index(&x.0, &y.0).then(x.1.cmp(&y.1)));
        a.dedup();
        a
    }

    /// Check if two layouts are equal in terms of their non-zero patterns.
    pub fn eq_nonzeros(&self, other: &Layout) -> bool {
        if self.kind != other.kind {
            return false;
        }
        if self.n_dense_axes != other.n_dense_axes {
            return false;
        }
        if self.shape != other.shape {
            return false;
        }
        if self.indices != other.indices {
            return false;
        }
        true
    }

    /// Check if two layouts are equal in terms of their non-zero patterns and dependencies.
    pub fn eq_nonzeros_and_deps(&self, other: &Layout) -> bool {
        if !self.eq_nonzeros(other) {
            return false;
        }
        if self.state_deps != other.state_deps {
            return false;
        }
        if self.input_deps != other.input_deps {
            return false;
        }
        true
    }

    /// Filter dependencies to only include those with indices in the provided iterator. The iterator will always
    /// be in the same sorted order as deps.
    fn filter_deps(
        new_deps_rank: usize,
        deps: Vec<NonZero>,
        filtered_indices: impl Iterator<Item = Index>,
    ) -> Vec<NonZero> {
        let mut filtered_deps = Vec::new();
        let mut dep_iter = deps.iter();
        let mut current_dep = dep_iter.next();
        for index in filtered_indices {
            while let Some((ref dep_index, dep_j)) = current_dep {
                match Self::cmp_index(dep_index, &index) {
                    std::cmp::Ordering::Less => {
                        current_dep = dep_iter.next();
                    }
                    std::cmp::Ordering::Greater => {
                        break;
                    }
                    std::cmp::Ordering::Equal => {
                        filtered_deps.push((index.slice(s![..new_deps_rank]).to_owned(), *dep_j));
                        current_dep = dep_iter.next();
                    }
                }
            }
        }
        filtered_deps
    }

    /// Remap dependency pairs from a source layout into a destination layout using an index mapping function.
    /// `map_fn` may return multiple destination indices for a single source index (e.g. broadcasting).
    fn remap_dependencies<F>(
        state_deps: &[NonZero],
        input_deps: &[NonZero],
        map_fn: F,
    ) -> (Vec<NonZero>, Vec<NonZero>)
    where
        F: Fn(&Index) -> Vec<Index>,
    {
        let remap = |deps: &[(Index, usize)]| -> Vec<(Index, usize)> {
            let mut remapped = Vec::new();
            for (src_index, dep_idx) in deps.iter() {
                for mapped_index in map_fn(src_index).into_iter() {
                    remapped.push((mapped_index, *dep_idx));
                }
            }
            remapped.sort_unstable_by(|x, y| Self::cmp_index(&x.0, &y.0).then(x.1.cmp(&y.1)));
            remapped.dedup();
            remapped
        };

        (remap(state_deps), remap(input_deps))
    }

    /// Convenience wrapper to remap this layout's dependencies onto a destination layout.
    fn remap_self_dependencies<F>(&self, map_fn: F) -> (Vec<NonZero>, Vec<NonZero>)
    where
        F: Fn(&Index) -> Vec<Index>,
    {
        Self::remap_dependencies(&self.state_deps, &self.input_deps, map_fn)
    }

    /// Pad or truncate an index to the required length, filling missing entries with zeros.
    fn fit_index_len(index: &Index, len: usize) -> Index {
        if index.len() == len {
            return index.clone();
        }
        let mut new_index = Index::zeros(len);
        for i in 0..std::cmp::min(index.len(), len) {
            new_index[i] = index[i];
        }
        new_index
    }

    /// Add state or input dependencies to this layout based on the tensor type.
    pub fn add_tensor_dependencies(&mut self, tensor_type: TensorType, start: i64, env: &mut Env) {
        let indices = match tensor_type {
            TensorType::State | TensorType::StateDot | TensorType::Input => {
                let mut deps = Vec::new();
                let n_states = *self.shape().get(0).unwrap_or(&1) as i64;
                for i in 0_i64..n_states {
                    let index = Index::from(vec![i]);
                    deps.push((index, (i + start) as usize));
                }
                deps
            }
            TensorType::Other => Vec::new(),
        };
        match tensor_type {
            TensorType::State => {
                assert!(
                    self.state_deps.is_empty(),
                    "state tensor layout should not already have state dependencies",
                );
                self.state_deps = indices;
                // store the state0 input dependencies in the env since we don't want to propagate them further
                env.state0_input_deps = mem::take(&mut self.input_deps);
            }
            TensorType::StateDot => {
                assert!(
                    self.state_deps.is_empty(),
                    "state dot tensor layout should not already have state dependencies",
                );
                self.state_deps = indices;
                // store the dstate0 input dependencies in the env since we don't want to propagate them further
                env.dstate0_input_deps = mem::take(&mut self.input_deps);
            }
            TensorType::Input => {
                assert! {
                    self.input_deps.is_empty(),
                    "input tensor layout should not already have input dependencies",
                };
                self.input_deps = indices;
            }
            TensorType::Other => {}
        }
    }

    // row major order
    pub fn unravel_index(index: usize, shape: &Shape) -> Index {
        let mut idx = index;
        let mut res = Index::zeros(shape.len());
        for i in (0..shape.len()).rev() {
            res[i] = i64::try_from(idx % shape[i]).unwrap();
            idx /= shape[i];
        }
        res
    }

    // row major order
    pub fn ravel_index(index: &Index, shape: &Shape) -> usize {
        let mut res = 0;
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            res += usize::try_from(index[i]).unwrap() * stride;
            stride *= shape[i];
        }
        res
    }

    // row major order
    pub fn cmp_index(a: &Index, b: &Index) -> std::cmp::Ordering {
        for i in 0..a.len() {
            match a[i].cmp(&b[i]) {
                std::cmp::Ordering::Less => return std::cmp::Ordering::Less,
                std::cmp::Ordering::Greater => return std::cmp::Ordering::Greater,
                _ => {}
            }
        }
        std::cmp::Ordering::Equal
    }

    // contract_last_axis contracts the last axis of the layout, returning a new layout with the last axis contracted.
    pub fn contract_last_axis(&self) -> Result<Layout> {
        let rank = self.rank();
        if rank == 0 {
            return Err(anyhow!("cannot contract last axis of a scalar"));
        }
        let new_shape = self.shape.slice(s![0..rank - 1]).to_owned();
        let (new_indices, new_kind, new_n_dense_axes) = if self.is_dense() {
            (Vec::new(), LayoutKind::Dense, new_shape.len())
        } else if self.is_diagonal() {
            if self.n_dense_axes == 0 && rank > 2 {
                (Vec::new(), LayoutKind::Diagonal, self.n_dense_axes)
            } else if self.n_dense_axes == 0 && rank <= 2 {
                (Vec::new(), LayoutKind::Dense, new_shape.len())
            } else {
                (Vec::new(), LayoutKind::Diagonal, self.n_dense_axes - 1)
            }
        } else if self.n_dense_axes == 0 {
            let mut new_indices = self.indices.clone();
            (0..self.indices.len()).for_each(|i| {
                new_indices[i] = new_indices[i].slice(s![0..rank - 1]).to_owned();
            });

            // remove any duplicate indices
            new_indices.sort_by(Self::cmp_index);
            new_indices.dedup();

            // check if now dense
            let new_nnz = new_shape.iter().product();
            if new_indices.len() == new_nnz {
                (new_indices, LayoutKind::Dense, self.n_dense_axes)
            } else {
                (new_indices, LayoutKind::Sparse, self.n_dense_axes)
            }
        } else {
            // there are dense axes, so just remove it
            (Vec::new(), LayoutKind::Sparse, self.n_dense_axes - 1)
        };

        let mut new_layout = Layout {
            indices: new_indices,
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: new_shape,
            kind: new_kind,
            n_dense_axes: new_n_dense_axes,
        };

        let (state_deps, input_deps) = self.remap_self_dependencies(|idx| {
            // If the stored index has one more dimension than the new layout rank, drop the last axis.
            let mut truncated = if idx.len() > new_layout.rank() {
                idx.slice(s![0..idx.len() - 1]).to_owned()
            } else {
                idx.clone()
            };
            truncated = Self::fit_index_len(&truncated, new_layout.rank());
            vec![truncated]
        });
        new_layout.state_deps = state_deps;
        new_layout.input_deps = input_deps;

        Ok(new_layout)
    }

    // permute the axes of the layout and return a new layout
    pub fn permute(&self, permutation: &[usize]) -> Result<Layout> {
        // check that we have the right number of permutation indices
        if permutation.len() > self.rank() {
            return Err(anyhow!("too many permutation indices"));
        }
        // check that permutation is a valid permutation
        if permutation.len() < self.min_rank() {
            return Err(anyhow!("not enough permutation indices"));
        }

        // if its an empty permutation then return the same layout
        if permutation.is_empty() {
            return Ok(self.clone());
        }

        let new_rank = permutation.iter().max().unwrap() + 1;

        // for a sparse tensor, can only permute the sparse axes
        if self.is_sparse() {
            #[allow(clippy::needless_range_loop)]
            for i in self.rank() - self.n_dense_axes..self.rank() {
                if permutation[i] != i {
                    return Err(anyhow!("cannot permute dense axes of a sparse layout"));
                }
            }
        }

        // permute shape
        let mut new_shape = Shape::ones(new_rank);
        for (i, &p) in permutation.iter().enumerate() {
            new_shape[p] = self.shape[i];
        }

        // permute indices
        let new_indices = self
            .indices
            .iter()
            .map(|i| {
                let mut new_i = Index::zeros(new_rank);
                for (pi, &p) in permutation.iter().enumerate() {
                    new_i[p] = i[pi];
                }
                new_i
            })
            .collect::<Vec<_>>();

        // reduce the number of dense axes according to the permutation
        let n_dense_axes = if self.is_dense() {
            new_rank
        } else {
            self.n_dense_axes
        };

        let mut new_layout = Layout {
            indices: new_indices,
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: new_shape,
            kind: self.kind.clone(),
            n_dense_axes,
        };

        let (state_deps, input_deps) = self.remap_self_dependencies(|idx| {
            let mut new_idx = Index::zeros(new_layout.rank());
            for (pi, &p) in permutation.iter().enumerate() {
                if pi < idx.len() {
                    new_idx[p] = idx[pi];
                }
            }
            vec![new_idx]
        });
        new_layout.state_deps = state_deps;
        new_layout.input_deps = input_deps;

        Ok(new_layout)
    }

    // create a new layout by broadcasting a list of layouts
    pub fn broadcast(layouts: Vec<Layout>, op: Option<char>) -> Result<Layout> {
        // the shapes of the layouts must be broadcastable
        let shapes = layouts.iter().map(|x| &x.shape).collect::<Vec<_>>();
        let shape = match broadcast_shapes(&shapes[..]) {
            Some(x) => x,
            None => {
                let shapes_str = shapes
                    .iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(anyhow!("cannot broadcast shapes: {}", shapes_str));
            }
        };

        let is_call = op.is_none();
        let is_divide = if let Some(op) = op { op == '/' } else { false };
        let is_multiply = if let Some(op) = op { op == '*' } else { false };
        let is_multiply_or_divide = is_multiply || is_divide;

        let mut broadcasted_layouts: Vec<Layout> = layouts
            .iter()
            .map(|l| l.broadcast_to_shape(&shape))
            .collect();

        let mut ret = broadcasted_layouts.pop().unwrap();
        if is_call {
            ret.to_dense();
        }
        let mut first = true;
        for layout in broadcasted_layouts.drain(..).rev() {
            // if a / b, with b is sparse and a being a dense or different sparse layout, then we have a divide by zero issue
            if first
                && is_divide
                && ret.is_sparse()
                && (layout.is_dense() || !ret.eq_nonzeros(&layout))
            {
                return Err(anyhow!("divide-by-zero detected, cannot only divide by a sparse layout if the numerator has the same sparsity pattern"));
            }
            if is_multiply_or_divide {
                ret = Self::intersect(ret, layout);
            } else if !is_call {
                ret = Self::union(ret, layout);
            } else {
                ret = Self::union_dense(ret, layout);
            }
            first = false;
        }

        // if now dense then convert to dense layout
        if ret.is_sparse_yet_dense() {
            let new_layout = Layout {
                indices: Vec::new(),
                state_deps: ret.state_deps,
                input_deps: ret.input_deps,
                n_dense_axes: ret.shape.len(),
                shape,
                kind: LayoutKind::Dense,
            };
            return Ok(new_layout);
        }

        // if now diagonal then convert to diagonal layout
        if ret.is_sparse_yet_diagonal() {
            let new_layout = Layout {
                indices: Vec::new(),
                state_deps: ret.state_deps,
                input_deps: ret.input_deps,
                n_dense_axes: ret.n_dense_axes,
                shape,
                kind: LayoutKind::Diagonal,
            };
            return Ok(new_layout);
        }
        Ok(ret)
    }
    pub fn is_dense(&self) -> bool {
        self.kind == LayoutKind::Dense
    }
    pub fn is_sparse(&self) -> bool {
        self.kind == LayoutKind::Sparse
    }
    pub fn is_diagonal(&self) -> bool {
        self.kind == LayoutKind::Diagonal
    }
    pub fn is_sparse_yet_dense(&self) -> bool {
        self.is_sparse() && self.indices.len() == self.shape.product()
    }

    pub fn is_sparse_yet_diagonal(&self) -> bool {
        if !self.is_sparse() {
            return false;
        }
        let is_vector = self.rank() == 1;
        let all_dims_equal = self.shape.iter().all(|&x| x == self.shape[0]);
        let num_indices_equal_to_dim = self.indices.len() == self.shape[0];
        let indices_not_equal = self
            .indices
            .iter()
            .any(|index| index.iter().any(|&x| x != index[0]));
        !is_vector && !indices_not_equal && all_dims_equal && num_indices_equal_to_dim
    }
    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn to_dense(&mut self) {
        if self.is_dense() {
            return;
        }
        self.indices = Vec::new();
        self.kind = LayoutKind::Dense;
        self.n_dense_axes = self.shape.len();
    }

    pub fn to_sparse(&mut self) {
        if self.is_sparse() {
            return;
        }
        let mut new_indices = Vec::new();
        match self.kind {
            LayoutKind::Dense => {
                for i in 0..self.shape.product() {
                    new_indices.push(Self::unravel_index(i, &self.shape));
                }
            }
            LayoutKind::Diagonal => {
                for i in 0..self.shape[0] {
                    let index = Index::zeros(self.rank()) + i64::try_from(i).unwrap();
                    new_indices.push(index);
                }
            }
            LayoutKind::Sparse => {
                unreachable!("already sparse");
            }
        }
        self.indices = new_indices;
        self.kind = LayoutKind::Sparse;
    }

    pub fn new_empty(rank: usize) -> Self {
        Layout {
            indices: vec![],
            state_deps: vec![],
            input_deps: vec![],
            shape: Shape::zeros(rank),
            kind: LayoutKind::Dense,
            n_dense_axes: rank,
        }
    }

    pub fn new_scalar() -> Self {
        Layout {
            indices: vec![],
            state_deps: vec![],
            input_deps: vec![],
            shape: Shape::zeros(0),
            kind: LayoutKind::Dense,
            n_dense_axes: 0,
        }
    }

    pub fn new_dense(shape: Shape) -> Self {
        let n_dense_axes = shape.len();
        Layout {
            indices: vec![],
            state_deps: vec![],
            input_deps: vec![],
            shape,
            kind: LayoutKind::Dense,
            n_dense_axes,
        }
    }

    pub fn new_diagonal(shape: Shape) -> Self {
        Layout {
            indices: vec![],
            state_deps: vec![],
            input_deps: vec![],
            shape,
            kind: LayoutKind::Diagonal,
            n_dense_axes: 0,
        }
    }

    pub fn new_diagonal_from(shape: Shape, layout: &Layout) -> Option<Self> {
        // should be scalar or vector layout, shape has only one non-one dimension
        if layout.is_scalar() {
            let mut state_deps = layout
                .state_deps
                .iter()
                .map(|(_idx, j)| (Index::zeros(shape.len()), *j))
                .collect::<Vec<_>>();
            let mut input_deps = layout
                .input_deps
                .iter()
                .map(|(_idx, j)| (Index::zeros(shape.len()), *j))
                .collect::<Vec<_>>();
            for i in 1..shape.len() {
                state_deps.extend(
                    layout
                        .state_deps
                        .iter()
                        .map(|(_idx, j)| (Index::zeros(shape.len()) + i as i64, *j)),
                );
                input_deps.extend(
                    layout
                        .input_deps
                        .iter()
                        .map(|(_idx, j)| (Index::zeros(shape.len()) + i as i64, *j)),
                );
            }
            return Some(Layout {
                indices: vec![],
                state_deps,
                input_deps,
                shape,
                kind: LayoutKind::Diagonal,
                n_dense_axes: 0,
            });
        }
        if layout.shape.iter().filter(|&&x| x > 1).count() > 1 {
            return None;
        }
        let axis = layout.shape.iter().position(|&x| x > 1).unwrap_or(0);

        let state_deps = layout
            .state_deps
            .iter()
            .map(|(idx, j)| (Index::zeros(shape.len()) + idx[axis], *j))
            .collect::<Vec<_>>();
        let input_deps = layout
            .input_deps
            .iter()
            .map(|(idx, j)| (Index::zeros(shape.len()) + idx[axis], *j))
            .collect::<Vec<_>>();
        Some(Layout {
            indices: vec![],
            state_deps,
            input_deps,
            shape,
            kind: LayoutKind::Diagonal,
            n_dense_axes: 0,
        })
    }

    /// filters the dependencies of layout, which is assumed to be a dense vector layout, to only include those in the range [start, end)
    /// subtracting start from the indices and assigning the filtered dependencies to self
    pub fn filter_deps_from(&mut self, mut layout: Layout, start: i64, end: i64) {
        assert!(layout.is_dense());
        assert!(layout.shape.iter().filter(|&&x| x > 1).count() <= 1);
        let axis = layout.shape.iter().position(|&x| x > 1).unwrap_or(0);
        // assert only one possible axis
        let layout_state_deps = mem::take(&mut layout.state_deps);
        let layout_input_deps = mem::take(&mut layout.input_deps);
        let indices = layout.indices().filter(|idx| {
            let i = idx[axis];
            i >= start && i < end
        });
        self.state_deps = Self::filter_deps(self.rank(), layout_state_deps, indices)
            .into_iter()
            .map(|(mut idx, j)| {
                if idx.len() > axis {
                    idx[axis] -= start;
                }
                (idx, j)
            })
            .collect();
        let indices = layout.indices().filter(|idx| {
            let i = idx[axis];
            i >= start && i < end
        });
        self.input_deps = Self::filter_deps(self.rank(), layout_input_deps, indices)
            .into_iter()
            .map(|(mut idx, j)| {
                if idx.len() > axis {
                    idx[axis] -= start;
                }
                (idx, j)
            })
            .collect();
    }

    // concatenate a list of layouts along the first axis
    pub fn concatenate(elmts: &[TensorBlock], rank: usize) -> Result<Self> {
        let layouts = elmts
            .iter()
            .map(|x| x.layout().as_ref())
            .collect::<Vec<_>>();
        let is_zero = elmts
            .iter()
            .map(|x| matches!(x.expr().kind, AstKind::Number(0.0)))
            .collect::<Vec<_>>();
        let starts = elmts.iter().map(|x| x.start()).collect::<Vec<_>>();

        // if there are no layouts then return an empty layout
        if layouts.is_empty() {
            return Ok(Layout::new_empty(0));
        }

        // get max rank of the elmts
        let max_rank = layouts.iter().map(|x| x.rank()).max().unwrap();
        if max_rank > rank {
            return Err(anyhow!(
                "cannot concatenate layouts with rank greater than the rank of the target tensor"
            ));
        }

        // get max shape of the elmts, each dim is at least 1
        let max_shape = layouts.iter().fold(Shape::ones(rank), |mut acc, x| {
            for i in 0..x.rank() {
                acc[i] = std::cmp::max(acc[i], x.shape()[i]);
            }
            acc
        });

        // check if the layouts are contiguous on the first axis
        // check if the layouts are contiguous on the diagonal
        let mut is_contiguous_on_first_axis = true;
        let mut is_contiguous_on_diagonal = true;
        if rank > 0 {
            let mut curr_index = 0;
            for (start, layout) in std::iter::zip(starts.iter(), layouts.iter()) {
                let mut expect_contiguous_on_first_axis = Index::zeros(rank);
                expect_contiguous_on_first_axis[0] = curr_index;
                is_contiguous_on_first_axis &= start == &expect_contiguous_on_first_axis;

                let expect_contiguous_on_diagonal = Index::zeros(rank) + curr_index;
                is_contiguous_on_diagonal &= start == &expect_contiguous_on_diagonal;

                curr_index += if layout.rank() == 0 {
                    1
                } else {
                    i64::try_from(layout.shape()[0]).unwrap()
                };
            }
        }

        let mut result_layout: Layout;

        // if all layouts are dense and contiguous then calculate the new shape
        if layouts.iter().all(|x| x.is_dense()) && is_contiguous_on_first_axis {
            // check that this shape can be broadcast into max_shape
            for layout in layouts.iter() {
                for i in 1..std::cmp::min(layout.rank(), rank) {
                    if layout.shape[i] != 1 && layout.shape[i] != max_shape[i] {
                        return Err(anyhow!(
                            "cannot concatenate layouts that cannot be broadcast into each other (got shapes {:?})",
                            layouts.iter().map(|x| x.shape()).collect::<Vec<_>>()
                        ));
                    }
                }
            }
            let mut new_shape = max_shape.clone();
            if rank > 0 {
                new_shape[0] = layouts
                    .iter()
                    .map(|x| if x.rank() > 0 { x.shape[0] } else { 1 })
                    .sum();
            }
            result_layout = Layout::dense(new_shape);
        } else {
            // must be at least one diagonal or sparse layout here, get its n_dense_axes
            let mut n_dense_axes = 0;
            for layout in layouts.iter() {
                if layout.is_diagonal() || layout.is_sparse() {
                    n_dense_axes = layout.n_dense_axes;
                    break;
                }
            }

            // check that the number of final dense axes is the same for all sparse or diagonal layouts
            if layouts
                .iter()
                .any(|x| !x.is_dense() && x.n_dense_axes != n_dense_axes)
            {
                return Err(anyhow!(
                    "cannot concatenate layouts with different numbers of final dense axes"
                ));
            }

            // check that the shapes of the final dense axes are the same for all layouts
            if layouts.iter().any(|x| {
                x.shape.slice(s![x.rank() - n_dense_axes..])
                    != layouts[0].shape.slice(s![x.rank() - n_dense_axes..])
            }) {
                return Err(anyhow!(
                    "cannot concatenate layouts with different shapes for the final dense axes"
                ));
            }

            // check that the rank is the same for all layouts
            if layouts.iter().any(|x| x.rank() != layouts[0].rank()) {
                return Err(anyhow!("cannot concatenate layouts with different ranks"));
            }

            // if all layouts are diagonal (or scalar), and on the diagonal then
            if layouts
                .iter()
                .all(|x| x.is_diagonal() || x.shape().iter().all(|x| *x == 1))
                && is_contiguous_on_diagonal
            {
                // add up the shapes for the non-final dense axes
                let mut new_shape = max_shape.clone();
                for i in 0..rank {
                    new_shape[i] = layouts.iter().map(|x| x.shape[i]).sum();
                }
                result_layout = Self {
                    shape: new_shape,
                    indices: Vec::new(),
                    state_deps: Vec::new(),
                    input_deps: Vec::new(),
                    kind: LayoutKind::Diagonal,
                    n_dense_axes,
                };
            } else {
                // find the maxiumum extent of the individual layouts
                let mut max_extent = Shape::zeros(rank);
                for (start, layout) in std::iter::zip(starts.iter(), layouts.iter()) {
                    for i in 0..layout.rank() {
                        max_extent[i] = std::cmp::max(
                            max_extent[i],
                            usize::try_from(start[i]).unwrap() + layout.shape[i],
                        );
                    }
                }

                // any other combination we convert all to sparse and concatenate
                let mut new_layout = Layout {
                    indices: Vec::new(),
                    state_deps: Vec::new(),
                    input_deps: Vec::new(),
                    shape: max_extent,
                    kind: LayoutKind::Sparse,
                    n_dense_axes,
                };
                for ((layout, start), is_zero) in
                    zip(zip(layouts.iter(), starts.iter()), is_zero.iter())
                {
                    if *is_zero {
                        continue;
                    }

                    // convert to sparse
                    new_layout.indices.extend(layout.indices().map(|mut x| {
                        for (i, xi) in x.iter_mut().enumerate() {
                            *xi += start[i];
                        }
                        x
                    }));
                }

                // sort the indices in row major order and remove duplicates
                new_layout.indices.sort_by(Self::cmp_index);
                new_layout.indices.dedup();

                // check if now dense
                if new_layout.indices.len() == new_layout.shape.product() {
                    new_layout.kind = LayoutKind::Dense;
                    new_layout.indices.clear();
                    new_layout.n_dense_axes = rank;
                }

                result_layout = new_layout;
            }
        }

        // collect dependencies from each source layout and offset indices by the block start
        let mut state_deps = Vec::new();
        let mut input_deps = Vec::new();
        for (layout, start) in std::iter::zip(layouts.iter(), starts.iter()) {
            let (s, i) = layout.remap_self_dependencies(|idx| {
                let mut new_idx = Layout::fit_index_len(idx, rank);
                for j in 0..std::cmp::min(new_idx.len(), start.len()) {
                    new_idx[j] += start[j];
                }
                vec![new_idx]
            });
            state_deps = Layout::merge_deps(state_deps, s);
            input_deps = Layout::merge_deps(input_deps, i);
        }
        result_layout.state_deps = state_deps;
        result_layout.input_deps = input_deps;
        Ok(result_layout)
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn min_rank(&self) -> usize {
        for i in (0..self.rank()).rev() {
            if self.shape[i] != 1 {
                return i + 1;
            }
        }
        0
    }

    pub fn dense(shape: Shape) -> Self {
        let n_dense_axes = shape.len();
        Self {
            indices: Vec::new(),
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape,
            kind: LayoutKind::Dense,
            n_dense_axes,
        }
    }
    pub fn diagonal(shape: Shape) -> Self {
        Self {
            indices: Vec::new(),
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape,
            kind: LayoutKind::Diagonal,
            n_dense_axes: 0,
        }
    }
    pub fn sparse(indices: Vec<Index>, shape: Shape) -> Self {
        Self {
            indices,
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape,
            kind: LayoutKind::Sparse,
            n_dense_axes: 0,
        }
    }

    /// returns the number of stored non-zero elements in the layout
    /// this does not include the implicit dense axes for sparse and diagonal layouts
    pub fn nnz(&self) -> usize {
        if self.is_dense() {
            self.shape.iter().product()
        } else if self.is_diagonal() {
            if self.shape.is_empty() {
                0
            } else {
                self.shape[0]
            }
        } else {
            self.indices.len()
        }
    }

    pub fn state_dependencies(&self) -> &Vec<(Index, usize)> {
        &self.state_deps
    }

    pub fn input_dependencies(&self) -> &Vec<(Index, usize)> {
        &self.input_deps
    }

    pub fn take_input_dependencies(&mut self) -> Vec<(Index, usize)> {
        std::mem::take(&mut self.input_deps)
    }

    /// return the non-zero indices of the layout as an iterator, corresponding to the order of the data entries
    pub fn indices(&self) -> impl Iterator<Item = Index> + '_ {
        match self.kind {
            LayoutKind::Dense => {
                let f = Box::new(move |i| Self::unravel_index(i, &self.shape));
                (0..self.shape.product()).map(f as Box<dyn Fn(usize) -> Index>)
            }
            LayoutKind::Diagonal => {
                let f = Box::new(move |i| Index::zeros(self.rank()) + i64::try_from(i).unwrap());
                (0..self.shape[0]).map(f as Box<dyn Fn(usize) -> Index>)
            }
            LayoutKind::Sparse => {
                let f = Box::new(move |i| {
                    let index: &Index = self.indices.get(i).unwrap();
                    index.clone()
                });
                (0..self.nnz()).map(f as Box<dyn Fn(usize) -> Index>)
            }
        }
    }

    /// return the explicit slice of indices for sparse layouts
    pub fn explicit_indices(&self) -> &[Index] {
        &self.indices
    }

    /// data entry for sparse block expression layouts
    pub fn to_data_layout(&self) -> Vec<i32> {
        let mut data_layout = vec![];
        if self.is_sparse() {
            for index in self.indices() {
                data_layout.extend(index.iter().map(|&x| x as i32));
            }
        }
        data_layout
    }

    // data entry when this layout is used within an expression with another layout
    // if the other layout is the same as self and permutation is [0,1,..n-1,n], then return an empty vec
    // if this layout is dense, then return an empty vec
    //
    // returns a vec with the same size as the number of nnz in other,
    // with each entry giving the index in self corresponding to that entry in other.
    // If an index in other does not exist in self, then a -1 is returned for that entry.
    // A permutation is also provided, giving the self index for each other index.
    pub fn to_binary_data_layout(&self, other: &Layout, permutation: &[usize]) -> Vec<i32> {
        if self.eq_nonzeros(other) && permutation.iter().enumerate().all(|(i, &p)| i == p) {
            return vec![];
        }
        if self.is_dense() {
            return vec![];
        }
        assert!(
            permutation.len() <= other.rank() + 1,
            "can have max one contracted axis"
        );
        let mut data_layout = vec![];
        let permute_index = |index: &Index| {
            let mut new_index = Index::zeros(self.rank());
            for (i, &p) in permutation.iter().enumerate() {
                if p < self.rank() {
                    new_index[p] = index[i];
                }
            }
            new_index
        };
        for index in other.indices() {
            let find_index = permute_index(&index);
            let nnz_index = self
                .find_nnz_index(&find_index)
                .map(|i| i as i32)
                .unwrap_or(-1);
            data_layout.push(nnz_index);
        }
        data_layout
    }

    /// broadcast this layout to the given rank, preserving the current shape
    /// if the rank is decreased, then we can only remove axes of size 1 from the end of the shape
    pub fn broadcast_to_rank(&self, rank: usize) -> Self {
        // new shape is the old shape with ones appended to the end
        let mut shape = Shape::ones(rank);
        for i in 0..min(self.rank(), rank) {
            shape[i] = self.shape[i];
        }
        // check the rest of the shape is ones
        for i in rank..self.rank() {
            assert_eq!(self.shape[i], 1);
        }
        self.broadcast_to_shape(&shape)
    }

    /// broadcast this layout to the given shape
    /// if we are increasing the rank, then we add dense axes
    /// if we are decreasing the rank, then we can only remove dense axes
    /// if any axis is being broadcasted, then it must be size 1 in the original layout
    pub fn broadcast_to_shape(&self, shape: &Shape) -> Self {
        for i in 0..min(self.rank(), shape.len()) {
            if self.shape[i] != shape[i] && self.shape[i] != 1 {
                panic!(
                    "cannot broadcast axis {} in layout shape {} to shape {}",
                    i, self.shape, shape
                );
            }
        }
        let mut indices = self.indices.clone();
        let broadcast_axes: Vec<usize> = (0..shape.len())
            .filter(|&i| i >= self.shape.len() || (self.shape[i] == 1 && self.shape[i] != shape[i]))
            .collect();
        let broadcast_sparse_axes = broadcast_axes
            .iter()
            .filter(|&&i| i < self.rank().saturating_sub(self.n_dense_axes))
            .cloned()
            .collect::<Vec<usize>>();

        // if sparse, we need to adjust the indices due to broadcasting
        if self.is_sparse() {
            for &axis in broadcast_sparse_axes.iter() {
                let mut new_broadcast_indices = Vec::new();
                for index in indices.iter() {
                    for j in 0..shape[axis] {
                        let mut new_bi = index.clone();
                        if axis < new_bi.len() {
                            new_bi[axis] = i64::try_from(j).unwrap();
                        }
                        new_broadcast_indices.push(new_bi);
                    }
                }
                indices = new_broadcast_indices;
            }
        }

        // sort the indices in standard ordering
        indices.sort_by(Self::cmp_index);

        // any layout needs to adjust its dependencies due to broadcasting
        let new_rank = shape.len();
        let (state_deps, input_deps) = self.remap_self_dependencies(|idx| {
            let mut base_idx = Index::zeros(new_rank);
            for (i, v) in idx.iter().enumerate() {
                if i < new_rank {
                    base_idx[i] = *v;
                }
            }
            let mut new_broadcast_indices = vec![base_idx.clone()];
            for &axis in broadcast_axes.iter() {
                for j in 0..shape[axis] {
                    let mut new_bi = base_idx.clone();
                    new_bi[axis] = i64::try_from(j).unwrap();
                    new_broadcast_indices.push(new_bi);
                }
            }
            new_broadcast_indices
        });

        if self.rank() == shape.len() {
            Self {
                indices,
                state_deps,
                input_deps,
                shape: shape.clone(),
                kind: self.kind.clone(),
                n_dense_axes: self.n_dense_axes,
            }
        } else if self.rank() < shape.len() {
            let new_ranks = shape.len() - self.rank();
            let n_dense_axes = self.n_dense_axes + new_ranks;
            Self {
                indices,
                state_deps,
                input_deps,
                shape: shape.clone(),
                kind: self.kind.clone(),
                n_dense_axes,
            }
        } else if (shape.len() < self.rank()) && (self.rank() - shape.len() <= self.n_dense_axes) {
            // must be reducing the rank by a number of dense axes
            let n_dense_axes = self.n_dense_axes - (self.rank() - shape.len());
            Self {
                indices,
                state_deps,
                input_deps,
                shape: shape.clone(),
                kind: self.kind.clone(),
                n_dense_axes,
            }
        } else {
            // invalid
            panic!(
                "cannot broadcast layout shape {} to shape {}. Requires removing more than {} dense axes.",
                self.shape,
                shape,
                self.n_dense_axes
            )
        }
    }

    // returns the index in the nnz array corresponding to the given index
    // this is broadcast aware, if the index is out of bounds for the layout shape and that axis is size 1, then the index is treated as 0
    // if the index has higher rank
    pub fn find_nnz_index(&self, index: &Index) -> Option<usize> {
        let bcast_index = {
            let mut new_index = index.clone();
            for i in 0..min(self.rank(), index.len()) {
                if self.shape[i] == 1 && index[i] > 0 {
                    new_index[i] = 0;
                } else if index[i] >= self.shape[i].try_into().unwrap() {
                    return None;
                }
            }
            new_index
        };
        // take off the dense axes for sparse and diagonal layouts
        let non_dense_index = bcast_index
            .slice(s![0..self.rank() - self.n_dense_axes])
            .to_owned();
        match self.kind {
            LayoutKind::Sparse => self
                .indices
                .binary_search_by(|x| Self::cmp_index(x, &non_dense_index))
                .ok(),
            LayoutKind::Dense => Some(Self::ravel_index(&bcast_index, self.shape())),
            LayoutKind::Diagonal => {
                if index.iter().all(|&x| x == non_dense_index[0])
                    && index[0] < self.shape[0].try_into().unwrap()
                {
                    Some(index[0].try_into().unwrap())
                } else {
                    None
                }
            }
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn kind(&self) -> &LayoutKind {
        &self.kind
    }

    pub fn n_dense_axes(&self) -> usize {
        self.n_dense_axes
    }

    /// only usable for sparse or dense layouts, adjusts n_dense_axes by expanding the indices
    /// can only reduce n_dense_axes
    fn remove_dense_axes(&mut self, new_n_dense_axes: usize) {
        assert!(
            self.is_sparse() || self.is_dense(),
            "can only remove dense axes from sparse or dense layouts"
        );
        assert!(new_n_dense_axes <= self.n_dense_axes);
        if self.n_dense_axes == new_n_dense_axes {
            return;
        }
        if self.is_dense() {
            self.n_dense_axes = new_n_dense_axes;
            return;
        }
        let rank = self.rank();
        let removed_axes = self.n_dense_axes - new_n_dense_axes;
        let dense_shape = self.shape.slice(s![rank - removed_axes..]).to_owned();
        let n_dense: usize = dense_shape.iter().product();
        let mut new_indices = Vec::new();
        for index in self.indices.iter() {
            for i in 0..n_dense {
                let mut new_index = Index::zeros(rank);
                for j in 0..std::cmp::min(index.len(), rank - removed_axes) {
                    new_index[j] = index[j];
                }
                let dense_index = Self::unravel_index(i, &dense_shape);
                for j in 0..removed_axes {
                    new_index[rank - removed_axes + j] = dense_index[j];
                }
                new_indices.push(new_index);
            }
        }

        let mut new_layout = Layout {
            indices: new_indices,
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: self.shape.clone(),
            kind: self.kind.clone(),
            n_dense_axes: new_n_dense_axes,
        };
        let (state_deps, input_deps) = self.remap_self_dependencies(|idx| {
            let mut mapped = Vec::new();
            for i in 0..n_dense {
                let dense_index = Layout::unravel_index(i, &dense_shape);
                let mut new_index = Index::zeros(rank);
                for j in 0..std::cmp::min(idx.len(), rank - removed_axes) {
                    new_index[j] = idx[j];
                }
                for j in 0..removed_axes {
                    new_index[rank - removed_axes + j] = dense_index[j];
                }
                mapped.push(new_index);
            }
            mapped
        });
        new_layout.state_deps = state_deps;
        new_layout.input_deps = input_deps;
        *self = new_layout;
    }

    /// the result is the dense union of the two layouts (i.e. from a call where f(0,0) != 0)
    /// result is always dense
    pub fn union_dense(self, other: Layout) -> Self {
        // union deps
        let state_deps = Layout::merge_deps(self.state_deps, other.state_deps);
        let input_deps = Layout::merge_deps(self.input_deps, other.input_deps);
        // result is always dense
        Layout {
            indices: Vec::new(),
            state_deps,
            input_deps,
            shape: self.shape.clone(),
            kind: LayoutKind::Dense,
            n_dense_axes: self.shape.len(),
        }
    }

    /// the result is the union of the two layouts (i.e. from addition/subtraction or any function where f(0,0) = 0
    ///
    /// 1. union deps
    /// 2. if either layout is dense, return dense layout
    /// 3. if both layouts are diagonal with same number of dense axes, return diagonal layout
    /// 4. no dense layouts, convert any diagonal layout to sparse
    /// 5. union indices, return the resulting sparse layout
    pub fn union(mut self, mut other: Layout) -> Self {
        assert!(
            self.shape == other.shape,
            "can only union layouts with the same shape"
        );

        // 1. union deps
        let state_deps = Layout::merge_deps(
            mem::take(&mut self.state_deps),
            mem::take(&mut other.state_deps),
        );
        let input_deps = Layout::merge_deps(
            mem::take(&mut self.input_deps),
            mem::take(&mut other.input_deps),
        );

        // if either layout is dense, return dense layout
        if self.is_dense() {
            self.state_deps = state_deps;
            self.input_deps = input_deps;
            return self;
        }

        if other.is_dense() {
            other.state_deps = state_deps;
            other.input_deps = input_deps;
            return other;
        }

        // if both layouts are diagonal with same number of dense axes, return diagonal layout
        if self.is_diagonal() && other.is_diagonal() && self.n_dense_axes == other.n_dense_axes {
            self.state_deps = state_deps;
            self.input_deps = input_deps;
            return self;
        }

        // convert any diagonal layout to sparse
        if self.is_diagonal() {
            self.to_sparse();
        }
        if other.is_diagonal() {
            other.to_sparse();
        }

        // union indices, return the resulting sparse layout
        if self.n_dense_axes > other.n_dense_axes {
            self.remove_dense_axes(other.n_dense_axes);
        } else if other.n_dense_axes > self.n_dense_axes {
            other.remove_dense_axes(self.n_dense_axes);
        }
        self.indices.extend(other.indices.iter().cloned());
        self.indices.sort_by(Self::cmp_index);
        self.indices.dedup();
        self.state_deps = state_deps;
        self.input_deps = input_deps;
        self
    }

    /// the result is the intersection of the two layouts (i.e. from multiplication)
    ///
    /// 1. start from sparsest layout (sparse, or self if both sparse)
    /// 2. intersect deps
    /// 3. if the remaining layout is dense, we're done
    /// 4. if both layouts are diagonal, we're done
    /// 5. if remaining layout is diagonal, convert it to sparse
    /// 6. must have two sparse layouts now, so intersect indices
    pub fn intersect(self, other: Layout) -> Self {
        assert!(
            self.shape == other.shape,
            "can only intersect layouts with the same shape"
        );

        // 1. start from sparsest layout (sparse, or self if both sparse)
        let mut sparse_layout: Layout;
        let mut other_layout: Layout;
        if self.is_sparse() {
            sparse_layout = self;
            other_layout = other;
        } else if other.is_sparse() {
            sparse_layout = other;
            other_layout = self;
        } else if self.is_diagonal() {
            sparse_layout = self;
            other_layout = other;
        } else if other.is_diagonal() {
            sparse_layout = other;
            other_layout = self;
        } else {
            sparse_layout = self;
            other_layout = other;
        }

        // 3. if the remaining layout is dense, we're done, return the sparse layout
        if other_layout.is_dense() {
            let state_deps = Layout::merge_deps(
                mem::take(&mut sparse_layout.state_deps),
                mem::take(&mut other_layout.state_deps),
            );
            let input_deps = Layout::merge_deps(
                mem::take(&mut sparse_layout.input_deps),
                mem::take(&mut other_layout.input_deps),
            );
            let state_deps =
                Layout::filter_deps(sparse_layout.rank(), state_deps, sparse_layout.indices());
            let input_deps =
                Layout::filter_deps(sparse_layout.rank(), input_deps, sparse_layout.indices());
            return Layout {
                indices: sparse_layout.indices.clone(),
                state_deps,
                input_deps,
                shape: sparse_layout.shape.clone(),
                kind: sparse_layout.kind.clone(),
                n_dense_axes: sparse_layout.n_dense_axes,
            };
        }

        // 4. if both layouts are diagonal with same number of dense axes, we're done
        if sparse_layout.is_diagonal()
            && other_layout.is_diagonal()
            && sparse_layout.n_dense_axes == other_layout.n_dense_axes
        {
            let state_deps = Layout::merge_deps(
                mem::take(&mut sparse_layout.state_deps),
                mem::take(&mut other_layout.state_deps),
            );
            let input_deps = Layout::merge_deps(
                mem::take(&mut sparse_layout.input_deps),
                mem::take(&mut other_layout.input_deps),
            );
            let n_dense_axes = sparse_layout.n_dense_axes;
            let ret = Layout {
                indices: Vec::new(),
                state_deps,
                input_deps,
                shape: sparse_layout.shape.clone(),
                kind: LayoutKind::Diagonal,
                n_dense_axes,
            };
            return ret;
        }

        // convert any diagonal layout to sparse
        if other_layout.is_diagonal() {
            other_layout.to_sparse();
        }
        if sparse_layout.is_diagonal() {
            sparse_layout.to_sparse();
        }

        // we must have two sparse layouts now, so intersect indices
        if sparse_layout.n_dense_axes > other_layout.n_dense_axes {
            sparse_layout.remove_dense_axes(other_layout.n_dense_axes);
        } else if other_layout.n_dense_axes > sparse_layout.n_dense_axes {
            other_layout.remove_dense_axes(sparse_layout.n_dense_axes);
        }
        let mut new_indices = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < sparse_layout.indices.len() && j < other_layout.indices.len() {
            match Self::cmp_index(&sparse_layout.indices[i], &other_layout.indices[j]) {
                std::cmp::Ordering::Less => {
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    new_indices.push(sparse_layout.indices[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }

        let mut ret = Layout {
            indices: new_indices,
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: sparse_layout.shape.clone(),
            kind: LayoutKind::Sparse,
            n_dense_axes: sparse_layout.n_dense_axes,
        };

        let state_deps = Layout::merge_deps(
            mem::take(&mut sparse_layout.state_deps),
            mem::take(&mut other_layout.state_deps),
        );
        let input_deps = Layout::merge_deps(
            mem::take(&mut sparse_layout.input_deps),
            mem::take(&mut other_layout.input_deps),
        );
        ret.state_deps = Layout::filter_deps(sparse_layout.rank(), state_deps, ret.indices());
        ret.input_deps = Layout::filter_deps(sparse_layout.rank(), input_deps, ret.indices());

        if ret.indices.len() == ret.shape.product() {
            ret.kind = LayoutKind::Dense;
            ret.indices.clear();
            ret.n_dense_axes = ret.shape.len();
        }
        ret
    }
}

// ArcLayout is a wrapper for Arc<Layout> that implements Hash and PartialEq based on ptr equality
#[derive(Debug)]
pub struct ArcLayout(Arc<Layout>);
impl Hash for ArcLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}
impl PartialEq for ArcLayout {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Clone for ArcLayout {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Eq for ArcLayout {}
impl Deref for ArcLayout {
    type Target = Layout;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl AsRef<Layout> for ArcLayout {
    fn as_ref(&self) -> &Layout {
        &self.0
    }
}
impl ArcLayout {
    pub fn new(layout: Layout) -> Self {
        Self(Arc::new(layout))
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_broadcast_layouts() -> Result<()> {
        let layout1 = Layout::dense(Shape::from(vec![2, 3, 4]));
        let layout2 = Layout::diagonal(Shape::from(vec![2, 3, 4]));
        let layout3 = Layout::sparse(vec![Index::from(vec![0, 0, 0])], Shape::from(vec![2, 3, 4]));
        let layout4 = Layout::sparse(
            vec![Index::from(vec![0, 0, 0]), Index::from(vec![1, 2, 3])],
            Shape::from(vec![2, 3, 4]),
        );
        let broadcasted1 = Layout::broadcast(vec![layout1.clone(), layout2.clone()], Some('*'))?;
        assert!(broadcasted1.is_diagonal());
        let broadcasted2 = Layout::broadcast(vec![layout1.clone(), layout2.clone()], Some('+'))?;
        assert!(broadcasted2.is_dense());
        let broadcasted3 = Layout::broadcast(vec![layout1.clone(), layout3.clone()], Some('*'))?;
        assert!(broadcasted3.is_sparse());
        let broadcasted4 = Layout::broadcast(vec![layout1.clone(), layout3.clone()], Some('+'))?;
        assert!(broadcasted4.is_dense());
        let broadcasted5 = Layout::broadcast(vec![layout3.clone(), layout4.clone()], Some('*'))?;
        assert!(broadcasted5.is_sparse());
        assert_eq!(broadcasted5.indices.len(), 1);
        assert_eq!(broadcasted5.indices[0], Index::from(vec![0, 0, 0]));
        let broadcasted6 = Layout::broadcast(vec![layout3.clone(), layout4.clone()], Some('+'))?;
        assert!(broadcasted6.is_sparse());
        assert_eq!(broadcasted6.indices.len(), 2);
        assert_eq!(broadcasted6.indices[0], Index::from(vec![0, 0, 0]));
        assert_eq!(broadcasted6.indices[1], Index::from(vec![1, 2, 3]));
        Ok(())
    }

    #[test]
    fn test_binary_data_layout() -> Result<()> {
        let layout1 = Layout::sparse(vec![Index::from(vec![0, 0, 0])], Shape::from(vec![2, 3, 4]));
        let layout2 = Layout::sparse(
            vec![Index::from(vec![0, 0, 0]), Index::from(vec![1, 2, 3])],
            Shape::from(vec![2, 3, 4]),
        );
        let layout1_plus_layout2 =
            Layout::broadcast(vec![layout1.clone(), layout2.clone()], Some('+'))?;
        assert!(layout1_plus_layout2.is_sparse());
        assert_eq!(layout1_plus_layout2.indices.len(), 2);
        assert_eq!(layout1_plus_layout2.indices[0], Index::from(vec![0, 0, 0]));
        assert_eq!(layout1_plus_layout2.indices[1], Index::from(vec![1, 2, 3]));
        let data_layout = layout1.to_binary_data_layout(&layout1_plus_layout2, &[0, 1, 2]);
        assert_eq!(data_layout, vec![0, -1]);
        let data_layout = layout2.to_binary_data_layout(&layout1_plus_layout2, &[0, 1, 2]);
        assert_eq!(data_layout, vec![]);
        Ok(())
    }

    #[test]
    fn test_union_sparse_layouts() {
        let layout1 = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        let layout3 = Layout::union(layout1, layout2);
        assert_eq!(layout3.indices.len(), 3);

        let mut layout1 = Layout::sparse(vec![Index::from(vec![1])], Shape::from(vec![2, 2]));
        layout1.n_dense_axes = 1;
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        let layout3 = Layout::union(layout1, layout2);
        assert_eq!(layout3.indices.len(), 2);
        assert_eq!(layout3.indices[0], Index::from(vec![1, 0]));
        assert_eq!(layout3.indices[1], Index::from(vec![1, 1]));
    }

    #[test]
    fn test_intersect_sparse_layouts() {
        let layout1 = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        let layout2 = Layout::sparse(
            vec![Index::from(vec![1, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        let layout3 = Layout::intersect(layout1, layout2);
        assert_eq!(layout3.indices.len(), 1);
        assert_eq!(layout3.indices[0], Index::from(vec![1, 1]));

        let mut layout1 = Layout::sparse(vec![Index::from(vec![1])], Shape::from(vec![2, 2]));
        layout1.n_dense_axes = 1;
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        let layout3 = Layout::intersect(layout1, layout2);
        assert_eq!(layout3.indices.len(), 1);
        assert_eq!(layout3.indices[0], Index::from(vec![1, 0]));
    }

    #[test]
    fn test_dependencies_roundtrip_dense_sparse() {
        let mut layout = Layout::sparse(
            vec![Index::from(vec![0]), Index::from(vec![1])],
            Shape::from(vec![2]),
        );
        layout.state_deps = vec![(Index::from(vec![1]), 7)];
        layout.input_deps = vec![(Index::from(vec![0]), 3)];

        layout.to_dense();
        assert!(layout.is_dense());
        assert_eq!(layout.state_deps, vec![(Index::from(vec![1]), 7)]);
        assert_eq!(layout.input_deps, vec![(Index::from(vec![0]), 3)]);

        layout.to_sparse();
        assert!(layout.is_sparse());
        assert_eq!(layout.state_deps, vec![(Index::from(vec![1]), 7)]);
        assert_eq!(layout.input_deps, vec![(Index::from(vec![0]), 3)]);
    }

    #[test]
    fn test_broadcast_to_shape_replicates_dependencies() {
        let mut layout = Layout::sparse(vec![Index::from(vec![0, 0])], Shape::from(vec![1, 2]));
        layout.state_deps = vec![(Index::from(vec![0, 0]), 42)];

        let broadcasted = layout.broadcast_to_shape(&Shape::from(vec![3, 2]));
        assert!(broadcasted.is_sparse());
        assert_eq!(broadcasted.shape, Shape::from(vec![3, 2]));
        assert_eq!(
            broadcasted.state_deps,
            vec![
                (Index::from(vec![0, 0]), 42),
                (Index::from(vec![1, 0]), 42),
                (Index::from(vec![2, 0]), 42)
            ]
        );
        assert!(broadcasted.input_deps.is_empty());
    }

    #[test]
    fn test_permute_remaps_dependencies() -> Result<()> {
        let mut layout = Layout::sparse(
            vec![Index::from(vec![0, 1]), Index::from(vec![1, 2])],
            Shape::from(vec![2, 3]),
        );
        layout.state_deps = vec![(Index::from(vec![1, 2]), 5)];
        layout.input_deps = vec![(Index::from(vec![0, 1]), 8)];

        let permuted = layout.permute(&[1, 0])?;
        assert_eq!(permuted.shape, Shape::from(vec![3, 2]));
        assert_eq!(
            permuted.indices,
            vec![Index::from(vec![1, 0]), Index::from(vec![2, 1])]
        );
        assert_eq!(permuted.state_deps, vec![(Index::from(vec![2, 1]), 5)]);
        assert_eq!(permuted.input_deps, vec![(Index::from(vec![1, 0]), 8)]);
        Ok(())
    }

    #[test]
    fn test_union_and_intersect_dependencies() {
        let mut layout1 = Layout::sparse(vec![Index::from(vec![0, 0])], Shape::from(vec![2, 2]));
        layout1.state_deps = vec![(Index::from(vec![0, 0]), 1)];

        let mut layout2 = Layout::sparse(vec![Index::from(vec![1, 1])], Shape::from(vec![2, 2]));
        layout2.input_deps = vec![(Index::from(vec![1, 1]), 2)];

        let union_layout = Layout::union(layout1, layout2);
        assert_eq!(
            union_layout.indices,
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])]
        );
        assert_eq!(union_layout.state_deps, vec![(Index::from(vec![0, 0]), 1)]);
        assert_eq!(union_layout.input_deps, vec![(Index::from(vec![1, 1]), 2)]);

        let mut layout3 = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        layout3.state_deps = vec![(Index::from(vec![1, 1]), 9)];
        layout3.input_deps = vec![(Index::from(vec![0, 0]), 8)];

        let mut layout4 = Layout::sparse(vec![Index::from(vec![1, 1])], Shape::from(vec![2, 2]));
        layout4.state_deps = vec![(Index::from(vec![1, 1]), 5)];

        let layout5 = Layout::intersect(layout3, layout4);
        assert_eq!(layout5.indices, vec![Index::from(vec![1, 1])]);
        assert_eq!(
            layout5.state_deps,
            vec![(Index::from(vec![1, 1]), 5), (Index::from(vec![1, 1]), 9)]
        );
        assert_eq!(layout5.input_deps, vec![]);
    }

    #[test]
    fn test_filter_deps_from_basic() {
        // Test basic filtering with a dense 1D layout
        let mut target_layout = Layout::dense(Shape::from(vec![5]));

        let mut source_layout = Layout::dense(Shape::from(vec![10]));
        source_layout.state_deps = vec![
            (Index::from(vec![0]), 0),
            (Index::from(vec![1]), 1),
            (Index::from(vec![2]), 2),
            (Index::from(vec![3]), 3),
            (Index::from(vec![4]), 4),
            (Index::from(vec![5]), 5),
            (Index::from(vec![6]), 6),
            (Index::from(vec![7]), 7),
        ];
        source_layout.input_deps = vec![
            (Index::from(vec![1]), 10),
            (Index::from(vec![3]), 11),
            (Index::from(vec![5]), 12),
            (Index::from(vec![7]), 13),
        ];

        // Filter to include indices 2..7
        target_layout.filter_deps_from(source_layout, 2, 7);

        // Should only include dependencies in the range [2, 7)
        assert_eq!(
            target_layout.state_deps,
            vec![
                (Index::from(vec![0]), 2),
                (Index::from(vec![1]), 3),
                (Index::from(vec![2]), 4),
                (Index::from(vec![3]), 5),
                (Index::from(vec![4]), 6),
            ]
        );
        assert_eq!(
            target_layout.input_deps,
            vec![(Index::from(vec![1]), 11), (Index::from(vec![3]), 12),]
        );
    }

    #[test]
    fn test_filter_deps_from_start_of_range() {
        // Test filtering from the start of the layout
        let mut target_layout = Layout::dense(Shape::from(vec![3]));

        let mut source_layout = Layout::dense(Shape::from(vec![8]));
        source_layout.state_deps = vec![
            (Index::from(vec![0]), 0),
            (Index::from(vec![1]), 1),
            (Index::from(vec![2]), 2),
            (Index::from(vec![5]), 5),
        ];
        source_layout.input_deps = vec![(Index::from(vec![0]), 10), (Index::from(vec![2]), 12)];

        // Filter to include indices 0..3
        target_layout.filter_deps_from(source_layout, 0, 3);

        assert_eq!(
            target_layout.state_deps,
            vec![
                (Index::from(vec![0]), 0),
                (Index::from(vec![1]), 1),
                (Index::from(vec![2]), 2),
            ]
        );
        assert_eq!(
            target_layout.input_deps,
            vec![(Index::from(vec![0]), 10), (Index::from(vec![2]), 12),]
        );
    }

    #[test]
    fn test_filter_deps_from_end_of_range() {
        // Test filtering from the end of the layout
        let mut target_layout = Layout::dense(Shape::from(vec![3]));

        let mut source_layout = Layout::dense(Shape::from(vec![8]));
        source_layout.state_deps = vec![
            (Index::from(vec![2]), 2),
            (Index::from(vec![5]), 5),
            (Index::from(vec![6]), 6),
            (Index::from(vec![7]), 7),
        ];
        source_layout.input_deps = vec![(Index::from(vec![5]), 15), (Index::from(vec![7]), 17)];

        // Filter to include indices 5..8
        target_layout.filter_deps_from(source_layout, 5, 8);

        assert_eq!(
            target_layout.state_deps,
            vec![
                (Index::from(vec![0]), 5),
                (Index::from(vec![1]), 6),
                (Index::from(vec![2]), 7),
            ]
        );
        assert_eq!(
            target_layout.input_deps,
            vec![(Index::from(vec![0]), 15), (Index::from(vec![2]), 17),]
        );
    }

    #[test]
    fn test_filter_deps_from_no_match_in_range() {
        // Test filtering when no dependencies fall in the range
        let mut target_layout = Layout::dense(Shape::from(vec![3]));

        let mut source_layout = Layout::dense(Shape::from(vec![10]));
        source_layout.state_deps = vec![
            (Index::from(vec![0]), 0),
            (Index::from(vec![1]), 1),
            (Index::from(vec![8]), 8),
            (Index::from(vec![9]), 9),
        ];
        source_layout.input_deps = vec![(Index::from(vec![0]), 10), (Index::from(vec![9]), 19)];

        // Filter to include indices 3..6 (no deps in this range)
        target_layout.filter_deps_from(source_layout, 3, 6);

        assert_eq!(target_layout.state_deps, vec![]);
        assert_eq!(target_layout.input_deps, vec![]);
    }

    #[test]
    fn test_filter_deps_from_single_index_range() {
        // Test filtering with a single index range
        let mut target_layout = Layout::dense(Shape::from(vec![1]));

        let mut source_layout = Layout::dense(Shape::from(vec![5]));
        source_layout.state_deps = vec![
            (Index::from(vec![0]), 0),
            (Index::from(vec![2]), 2),
            (Index::from(vec![3]), 3),
        ];
        source_layout.input_deps = vec![(Index::from(vec![2]), 12)];

        // Filter to include only index 2
        target_layout.filter_deps_from(source_layout, 2, 3);

        assert_eq!(target_layout.state_deps, vec![(Index::from(vec![0]), 2)]);
        assert_eq!(target_layout.input_deps, vec![(Index::from(vec![0]), 12)]);
    }

    #[test]
    fn test_filter_deps_from_empty_source() {
        // Test filtering when source has no dependencies
        let mut target_layout = Layout::dense(Shape::from(vec![5]));

        let source_layout = Layout::dense(Shape::from(vec![10]));
        // source_layout has no dependencies

        target_layout.filter_deps_from(source_layout, 2, 7);

        assert_eq!(target_layout.state_deps, vec![]);
        assert_eq!(target_layout.input_deps, vec![]);
    }

    #[test]
    fn test_filter_deps_from_full_range() {
        // Test filtering with the full range of the source layout
        let mut target_layout = Layout::dense(Shape::from(vec![5]));

        let mut source_layout = Layout::dense(Shape::from(vec![5]));
        source_layout.state_deps = vec![
            (Index::from(vec![0]), 0),
            (Index::from(vec![1]), 1),
            (Index::from(vec![2]), 2),
            (Index::from(vec![3]), 3),
            (Index::from(vec![4]), 4),
        ];
        source_layout.input_deps = vec![(Index::from(vec![1]), 11), (Index::from(vec![3]), 13)];

        // Filter to include all indices 0..5
        target_layout.filter_deps_from(source_layout, 0, 5);

        assert_eq!(
            target_layout.state_deps,
            vec![
                (Index::from(vec![0]), 0),
                (Index::from(vec![1]), 1),
                (Index::from(vec![2]), 2),
                (Index::from(vec![3]), 3),
                (Index::from(vec![4]), 4),
            ]
        );
        assert_eq!(
            target_layout.input_deps,
            vec![(Index::from(vec![1]), 11), (Index::from(vec![3]), 13),]
        );
    }

    #[test]
    fn test_broadcast_sparse_2d_with_dense_1d_state_deps() -> Result<()> {
        // Create a sparse 2D layout
        let sparse_2d = Layout::sparse(
            vec![
                Index::from(vec![0, 0]),
                Index::from(vec![0, 1]),
                Index::from(vec![1, 1]),
            ],
            Shape::from(vec![2, 2]),
        );

        // Create a dense 1D layout with state dependencies
        let mut dense_1d = Layout::dense(Shape::from(vec![2]));
        dense_1d.state_deps = vec![(Index::from(vec![0]), 0), (Index::from(vec![1]), 1)];

        // Broadcast with addition operation
        let result = Layout::broadcast(vec![sparse_2d.clone(), dense_1d.clone()], Some('+'))?;

        // Result should be dense when broadcasting with a dense layout
        assert!(result.is_dense());
        assert_eq!(result.shape(), &Shape::from(vec![2, 2]));

        // Check state dependencies are propagated correctly
        // The dense 1D state_deps are broadcast along the first axis only
        assert_eq!(
            result.state_deps,
            vec![
                (Index::from(vec![0, 0]), 0),
                (Index::from(vec![0, 1]), 0),
                (Index::from(vec![1, 0]), 1),
                (Index::from(vec![1, 1]), 1),
            ]
        );
        assert_eq!(result.input_deps, vec![]);

        Ok(())
    }

    #[test]
    fn test_broadcast_sparse_2d_with_dense_1d_multiply() -> Result<()> {
        // Create a sparse 2D layout
        let sparse_2d = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );

        // Create a dense 1D layout with state dependencies
        let mut dense_1d = Layout::dense(Shape::from(vec![2]));
        dense_1d.state_deps = vec![(Index::from(vec![0]), 0), (Index::from(vec![1]), 1)];

        // Broadcast with multiplication operation
        let result = Layout::broadcast(vec![sparse_2d, dense_1d], Some('*'))?;

        // When multiplying diagonal sparse with dense, result is diagonal
        assert!(result.is_diagonal());
        assert_eq!(result.shape(), &Shape::from(vec![2, 2]));

        // State dependencies only at diagonal positions (off-diagonal zeros eliminate dependencies)
        assert_eq!(
            result.state_deps,
            vec![(Index::from(vec![0, 0]), 0), (Index::from(vec![1, 1]), 1),]
        );
        assert_eq!(result.input_deps, vec![]);

        Ok(())
    }

    #[test]
    fn test_broadcast_sparse_2d_with_dense_1d_mixed_deps() -> Result<()> {
        // Create a sparse 2D layout with its own dependencies
        let mut sparse_2d = Layout::sparse(
            vec![Index::from(vec![0, 1]), Index::from(vec![1, 0])],
            Shape::from(vec![2, 2]),
        );
        sparse_2d.input_deps = vec![(Index::from(vec![0, 1]), 5), (Index::from(vec![1, 0]), 6)];

        // Create a dense 1D layout with state dependencies
        let mut dense_1d = Layout::dense(Shape::from(vec![2]));
        dense_1d.state_deps = vec![(Index::from(vec![0]), 0), (Index::from(vec![1]), 1)];

        // Broadcast with addition
        let result = Layout::broadcast(vec![sparse_2d.clone(), dense_1d.clone()], Some('+'))?;

        // Result should be dense when one operand is dense
        assert!(result.is_dense());

        // State deps from dense_1d broadcast along first axis to all columns
        assert_eq!(
            result.state_deps,
            vec![
                (Index::from(vec![0, 0]), 0),
                (Index::from(vec![0, 1]), 0),
                (Index::from(vec![1, 0]), 1),
                (Index::from(vec![1, 1]), 1),
            ]
        );

        // Input deps from sparse_2d are broadcast to matching positions
        assert_eq!(
            result.input_deps,
            vec![(Index::from(vec![0, 1]), 5), (Index::from(vec![1, 0]), 6),]
        );

        Ok(())
    }

    #[test]
    fn test_broadcast_1d_to_2d_dense_deps() {
        // Test broadcasting a 1D dense layout with dependencies to 2D
        let mut layout_1d = Layout::dense(Shape::from(vec![2]));
        layout_1d.state_deps = vec![(Index::from(vec![0]), 0), (Index::from(vec![1]), 1)];

        let layout_2d = layout_1d.broadcast_to_shape(&Shape::from(vec![2, 2]));

        assert!(layout_2d.is_dense());
        assert_eq!(layout_2d.shape(), &Shape::from(vec![2, 2]));

        // Dependencies should be replicated across the new dimension
        assert_eq!(
            layout_2d.state_deps,
            vec![
                (Index::from(vec![0, 0]), 0),
                (Index::from(vec![0, 1]), 0),
                (Index::from(vec![1, 0]), 1),
                (Index::from(vec![1, 1]), 1),
            ]
        );
    }

    #[test]
    fn test_1d_no_broadcast_preserves_deps() {
        // Test that a 1D layout used without broadcasting keeps its dependencies
        let mut layout_1d = Layout::dense(Shape::from(vec![2]));
        layout_1d.state_deps = vec![(Index::from(vec![0]), 0), (Index::from(vec![1]), 1)];

        // Using the same shape (no broadcast) should not change dependencies
        let same_layout = layout_1d.broadcast_to_shape(&Shape::from(vec![2]));

        assert!(same_layout.is_dense());
        assert_eq!(same_layout.shape(), &Shape::from(vec![2]));

        // Dependencies should remain 1D
        assert_eq!(
            same_layout.state_deps,
            vec![(Index::from(vec![0]), 0), (Index::from(vec![1]), 1),]
        );
    }
}
