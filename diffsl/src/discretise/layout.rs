use anyhow::{anyhow, Result};
use ndarray::s;
use std::{
    cmp::min,
    convert::AsRef,
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::Arc,
};

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
    Input,
    Other,
}

type NonZero = (Index, usize);

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
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    indices: Vec<Index>,
    state_deps: Vec<(Index, usize)>,
    input_deps: Vec<(Index, usize)>,
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
    pub fn add_tensor_dependencies(&mut self, tensor_type: TensorType) {
        let indices = match tensor_type {
            TensorType::State | TensorType::Input => {
                let mut deps = Vec::new();
                let n_states = *self.shape().get(0).unwrap_or(&1);
                for i in 0..n_states {
                    let index = Index::from(vec![i as i64]);
                    deps.push((index, i));
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
    pub fn broadcast(mut layouts: Vec<Layout>, op: Option<char>) -> Result<Layout> {
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

        let all_dense = layouts.iter().all(|x| x.is_dense());
        let is_call = op.is_none();

        if all_dense || is_call {
            let mut broadcasted: Vec<Layout> = layouts
                .iter()
                .map(|l| l.broadcast_to_shape(&shape))
                .collect();
            let mut ret = Layout::dense(shape);
            let mut state_deps = Vec::new();
            let mut input_deps = Vec::new();
            for layout in broadcasted.drain(..) {
                let (s, i) = layout
                    .remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
                state_deps = Layout::merge_deps(state_deps, s);
                input_deps = Layout::merge_deps(input_deps, i);
            }
            ret.state_deps = state_deps;
            ret.input_deps = input_deps;
            return Ok(ret);
        }

        let any_sparse = layouts.iter().any(|x| x.is_sparse());
        let any_diagonal = layouts.iter().any(|x| x.is_diagonal());
        let all_diagonal = layouts.iter().all(|x| x.is_diagonal());

        let is_divide = if let Some(op) = op { op == '/' } else { false };
        let is_multiply = if let Some(op) = op { op == '*' } else { false };
        let is_multiply_or_divide = is_multiply || is_divide;

        // special diagonal handling
        // case a: all diagonal layouts with the name number of dense axes -> diagonal layout
        // case b: multiply and all diagonal with the same number of dense axes and dense layouts -> diagonal layout
        // otherwise convert diagonal layouts to sparse layouts and continue
        if any_diagonal {
            let mut n_dense_axes = None;
            let mut all_same_dense_axes = true;
            for layout in layouts.iter() {
                if layout.is_diagonal() {
                    #[allow(clippy::unnecessary_unwrap)]
                    if n_dense_axes.is_none() {
                        n_dense_axes = Some(layout.n_dense_axes);
                    } else if layout.n_dense_axes != n_dense_axes.unwrap() {
                        all_same_dense_axes = false;
                    }
                }
            }
            let n_dense_axes = n_dense_axes.unwrap();

            if all_diagonal && all_same_dense_axes {
                let mut ret = Layout {
                    indices: Vec::new(),
                    state_deps: Vec::new(),
                    input_deps: Vec::new(),
                    shape,
                    kind: LayoutKind::Diagonal,
                    n_dense_axes,
                };
                let broadcasted: Vec<Layout> = layouts
                    .iter()
                    .map(|l| l.broadcast_to_shape(&ret.shape))
                    .collect();
                let mut state_deps = Vec::new();
                let mut input_deps = Vec::new();
                for layout in broadcasted.iter() {
                    let (s, i) = layout.remap_self_dependencies(|idx| {
                        vec![Layout::fit_index_len(idx, ret.rank())]
                    });
                    state_deps = Layout::merge_deps(state_deps, s);
                    input_deps = Layout::merge_deps(input_deps, i);
                }
                ret.state_deps = state_deps;
                ret.input_deps = input_deps;
                return Ok(ret);
            }

            if is_multiply && all_same_dense_axes && !any_sparse {
                let mut ret = Layout {
                    indices: Vec::new(),
                    state_deps: Vec::new(),
                    input_deps: Vec::new(),
                    shape,
                    kind: LayoutKind::Diagonal,
                    n_dense_axes,
                };
                let broadcasted: Vec<Layout> = layouts
                    .iter()
                    .map(|l| l.broadcast_to_shape(&ret.shape))
                    .collect();
                let mut state_deps = Vec::new();
                let mut input_deps = Vec::new();
                for layout in broadcasted.iter() {
                    let (s, i) = layout.remap_self_dependencies(|idx| {
                        vec![Layout::fit_index_len(idx, ret.rank())]
                    });
                    state_deps = Layout::merge_deps(state_deps, s);
                    input_deps = Layout::merge_deps(input_deps, i);
                }
                ret.state_deps = state_deps;
                ret.input_deps = input_deps;
                return Ok(ret);
            }

            // convert diagonal layouts to sparse layouts
            for layout in layouts.iter_mut() {
                if layout.is_diagonal() {
                    layout.to_sparse();
                }
            }
        }

        // if there are any sparse layouts then the result is sparse,
        // and the indicies of all sparse layouts must be identical and have the same number of dense axis.
        // must be sparse and maybe dense
        //
        let mut broadcasted_layouts: Vec<Layout> = layouts
            .into_iter()
            .map(|l| l.broadcast_to_shape(&shape))
            .collect();
        let mut ret = broadcasted_layouts.pop().unwrap();
        let mut dep_sources = vec![ret.clone()];
        let mut first = true;
        for layout in broadcasted_layouts.drain(..).rev() {
            dep_sources.push(layout.clone());

            // if a / b, with b is sparse and a being a dense or different sparse layout, then we have a divide by zero issue
            if first && is_divide && ret.is_sparse() && (layout.is_dense() || !ret.eq(&layout)) {
                return Err(anyhow!("divide-by-zero detected, cannot only divide by a sparse layout if the numerator has the same sparsity pattern"));
            }
            if is_multiply_or_divide {
                ret.intersect_inplace(layout);
            } else {
                ret.union_inplace(layout);
            }
            first = false;
        }

        // if now dense then convert to dense layout
        if ret.is_sparse_yet_dense() {
            let mut new_layout = Layout {
                indices: Vec::new(),
                state_deps: Vec::new(),
                input_deps: Vec::new(),
                n_dense_axes: ret.shape.len(),
                shape,
                kind: LayoutKind::Dense,
            };
            let mut state_deps = Vec::new();
            let mut input_deps = Vec::new();
            for layout in dep_sources.iter() {
                let (s, i) = layout.remap_self_dependencies(|idx| {
                    vec![Layout::fit_index_len(idx, new_layout.rank())]
                });
                state_deps = Layout::merge_deps(state_deps, s);
                input_deps = Layout::merge_deps(input_deps, i);
            }
            new_layout.state_deps = state_deps;
            new_layout.input_deps = input_deps;
            return Ok(new_layout);
        }

        // if now diagonal then convert to diagonal layout
        if ret.is_sparse_yet_diagonal() {
            let mut new_layout = Layout {
                indices: Vec::new(),
                state_deps: Vec::new(),
                input_deps: Vec::new(),
                n_dense_axes: ret.n_dense_axes,
                shape,
                kind: LayoutKind::Diagonal,
            };
            let mut state_deps = Vec::new();
            let mut input_deps = Vec::new();
            for layout in dep_sources.iter() {
                let (s, i) = layout.remap_self_dependencies(|idx| {
                    vec![Layout::fit_index_len(idx, new_layout.rank())]
                });
                state_deps = Layout::merge_deps(state_deps, s);
                input_deps = Layout::merge_deps(input_deps, i);
            }
            new_layout.state_deps = state_deps;
            new_layout.input_deps = input_deps;
            return Ok(new_layout);
        }
        // propagate dependencies for the general sparse case
        let mut state_deps = Vec::new();
        let mut input_deps = Vec::new();
        for layout in dep_sources.iter() {
            let (s, i) =
                layout.remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
            state_deps = Layout::merge_deps(state_deps, s);
            input_deps = Layout::merge_deps(input_deps, i);
        }
        ret.state_deps = state_deps;
        ret.input_deps = input_deps;
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
        let mut new_layout = Layout {
            indices: Vec::new(),
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: self.shape.clone(),
            kind: LayoutKind::Dense,
            n_dense_axes: self.shape.len(),
        };
        let (state_deps, input_deps) =
            self.remap_self_dependencies(|idx| vec![Self::fit_index_len(idx, new_layout.rank())]);
        new_layout.state_deps = state_deps;
        new_layout.input_deps = input_deps;
        *self = new_layout;
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
        let mut new_layout = Layout {
            indices: new_indices,
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: self.shape.clone(),
            kind: LayoutKind::Sparse,
            n_dense_axes: self.n_dense_axes,
        };
        let (state_deps, input_deps) =
            self.remap_self_dependencies(|idx| vec![Self::fit_index_len(idx, new_layout.rank())]);
        new_layout.state_deps = state_deps;
        new_layout.input_deps = input_deps;
        *self = new_layout;
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

    // concatenate a list of layouts along the first axis
    pub fn concatenate(elmts: &[TensorBlock], rank: usize) -> Result<Self> {
        let layouts = elmts
            .iter()
            .map(|x| x.layout().as_ref())
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
                for (layout, start) in std::iter::zip(layouts.iter(), starts.iter()) {
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
    // if the other layout is the same as self, then return an empty vec
    // if this layout is dense or diagonal, then return an empty vec
    //
    // returns a vec with the same size as the number of nnz in other,
    // with each entry giving the index in self corresponding to that entry in other.
    // If an index in other does not exist in self, then a -1 is returned for that entry.
    // A permutation is also provided, giving the self index for each other index.
    pub fn to_binary_data_layout(&self, other: &Layout, permutation: &[usize]) -> Vec<i32> {
        if self == other {
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
        let broadcast_axes: Vec<usize> = (0..self.rank().saturating_sub(self.n_dense_axes))
            .filter(|&i| self.shape[i] == 1 && self.shape[i] != shape[i])
            .collect();

        // if sparse, we need to adjust the indices due to broadcasting
        if self.is_sparse() {
            for &axis in broadcast_axes.iter() {
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

        let mut new_layout = if self.rank() == shape.len() {
            Self {
                indices,
                state_deps: Vec::new(),
                input_deps: Vec::new(),
                shape: shape.clone(),
                kind: self.kind.clone(),
                n_dense_axes: self.n_dense_axes,
            }
        } else if self.rank() < shape.len() {
            let new_ranks = shape.len() - self.rank();
            let n_dense_axes = self.n_dense_axes + new_ranks;
            Self {
                indices,
                state_deps: Vec::new(),
                input_deps: Vec::new(),
                shape: shape.clone(),
                kind: self.kind.clone(),
                n_dense_axes,
            }
        } else if (shape.len() < self.rank()) && (self.rank() - shape.len() <= self.n_dense_axes) {
            // must be reducing the rank by a number of dense axes
            let n_dense_axes = self.n_dense_axes - (self.rank() - shape.len());
            Self {
                indices,
                state_deps: Vec::new(),
                input_deps: Vec::new(),
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
        };

        let new_rank = new_layout.rank();
        let (state_deps, input_deps) = self.remap_self_dependencies(|idx| {
            let mut base = Index::zeros(new_rank);
            for i in 0..std::cmp::min(idx.len(), base.len()) {
                // when broadcasting axis size 1, default to zero
                base[i] = if self.shape[i] == 1 { 0 } else { idx[i] };
            }
            let mut seeds = vec![base];
            for &axis in broadcast_axes.iter() {
                // only broadcast if the axis still exists in the new rank
                if axis >= new_rank {
                    continue;
                }
                let mut next = Vec::new();
                for seed in seeds.iter() {
                    for j in 0..shape[axis] {
                        let mut new_seed = seed.clone();
                        new_seed[axis] = i64::try_from(j).unwrap();
                        next.push(new_seed);
                    }
                }
                seeds = next;
            }
            // if rank has increased, pad the remaining axes with zeros
            for seed in seeds.iter_mut() {
                if seed.len() < new_rank {
                    *seed = Layout::fit_index_len(seed, new_rank);
                }
            }
            if seeds.is_empty() {
                vec![Index::zeros(new_rank)]
            } else {
                seeds
            }
        });
        new_layout.state_deps = state_deps;
        new_layout.input_deps = input_deps;
        new_layout
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
            LayoutKind::Sparse => self.indices.iter().position(|x| x == non_dense_index),
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

    /// both self and other should be either dense or sparse layouts with the same shape
    /// the result is the union of the two layouts
    /// Note: one of teh layouts could have a different number of dense axes, in which case
    /// the dense axes are removed from the layout with more dense axes
    pub fn union_inplace(&mut self, mut other: Layout) {
        assert!(
            self.is_sparse() || self.is_dense(),
            "can only union sparse or dense layouts"
        );
        assert!(
            other.is_sparse() || other.is_dense(),
            "can only union sparse or dense layouts"
        );
        assert!(
            self.shape == other.shape,
            "can only union layouts with the same shape"
        );

        // union with a dense layout results in a dense layout
        if self.is_dense() || other.is_dense() {
            let mut ret = Layout::dense(self.shape.clone());
            let (s_self, i_self) =
                self.remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
            let (s_other, i_other) =
                other.remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
            ret.state_deps = Layout::merge_deps(s_self, s_other);
            ret.input_deps = Layout::merge_deps(i_self, i_other);
            *self = ret;
            return;
        }

        if self.n_dense_axes > other.n_dense_axes {
            self.remove_dense_axes(other.n_dense_axes);
        } else if other.n_dense_axes > self.n_dense_axes {
            other.remove_dense_axes(self.n_dense_axes);
        }
        self.indices.extend(other.indices.iter().cloned());
        self.indices.sort_by(Self::cmp_index);
        self.indices.dedup();

        // compute resulting dependencies
        let mut ret = Layout {
            indices: self.indices.clone(),
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: self.shape.clone(),
            kind: LayoutKind::Sparse,
            n_dense_axes: self.n_dense_axes,
        };
        // check if now dense
        if ret.indices.len() == ret.shape.product() {
            ret.kind = LayoutKind::Dense;
            ret.indices.clear();
            ret.n_dense_axes = ret.shape.len();
        }

        let (s_self, i_self) =
            self.remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
        let (s_other, i_other) =
            other.remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
        ret.state_deps = Layout::merge_deps(s_self, s_other);
        ret.input_deps = Layout::merge_deps(i_self, i_other);

        *self = ret;
    }

    /// self or other is either a sparse or dense layout with the same shape and n_dense_axes
    /// the result is the intersection of the two layouts
    pub fn intersect_inplace(&mut self, mut other: Layout) {
        assert!(
            self.is_sparse() || self.is_dense(),
            "can only intersect sparse or dense layouts"
        );
        assert!(
            other.is_sparse() || other.is_dense(),
            "can only intersect sparse or dense layouts"
        );
        assert!(
            self.shape == other.shape,
            "can only intersect layouts with the same shape"
        );

        if other.is_dense() {
            return;
        }
        if self.is_dense() {
            // result is simply the other layout
            *self = other;
            return;
        }
        if self.n_dense_axes > other.n_dense_axes {
            self.remove_dense_axes(other.n_dense_axes);
        } else if other.n_dense_axes > self.n_dense_axes {
            other.remove_dense_axes(self.n_dense_axes);
        }
        let mut new_indices = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < self.indices.len() && j < other.indices.len() {
            match Self::cmp_index(&self.indices[i], &other.indices[j]) {
                std::cmp::Ordering::Less => {
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    new_indices.push(self.indices[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }
        self.indices = new_indices;

        let mut ret = Layout {
            indices: self.indices.clone(),
            state_deps: Vec::new(),
            input_deps: Vec::new(),
            shape: self.shape.clone(),
            kind: LayoutKind::Sparse,
            n_dense_axes: self.n_dense_axes,
        };

        if ret.indices.len() == ret.shape.product() {
            ret.kind = LayoutKind::Dense;
            ret.indices.clear();
            ret.n_dense_axes = ret.shape.len();
        }

        let (s_self, i_self) =
            self.remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
        let (s_other, i_other) =
            other.remap_self_dependencies(|idx| vec![Layout::fit_index_len(idx, ret.rank())]);
        ret.state_deps = Layout::merge_deps(s_self, s_other);
        ret.input_deps = Layout::merge_deps(i_self, i_other);

        *self = ret;
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
        let mut layout1 = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        layout1.union_inplace(layout2);
        assert_eq!(layout1.indices.len(), 3);

        let mut layout1 = Layout::sparse(vec![Index::from(vec![1])], Shape::from(vec![2, 2]));
        layout1.n_dense_axes = 1;
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        layout1.union_inplace(layout2);
        assert_eq!(layout1.indices.len(), 2);
        assert_eq!(layout1.indices[0], Index::from(vec![1, 0]));
        assert_eq!(layout1.indices[1], Index::from(vec![1, 1]));
    }

    #[test]
    fn test_intersect_sparse_layouts() {
        let mut layout1 = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        let layout2 = Layout::sparse(
            vec![Index::from(vec![1, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        layout1.intersect_inplace(layout2);
        assert_eq!(layout1.indices.len(), 1);
        assert_eq!(layout1.indices[0], Index::from(vec![1, 1]));

        let mut layout1 = Layout::sparse(vec![Index::from(vec![1])], Shape::from(vec![2, 2]));
        layout1.n_dense_axes = 1;
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        layout1.intersect_inplace(layout2);
        assert_eq!(layout1.indices.len(), 1);
        assert_eq!(layout1.indices[0], Index::from(vec![1, 0]));
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

        let mut union_layout = layout1.clone();
        union_layout.union_inplace(layout2.clone());
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

        layout3.intersect_inplace(layout4);
        assert_eq!(layout3.indices, vec![Index::from(vec![1, 1])]);
        assert_eq!(
            layout3.state_deps,
            vec![(Index::from(vec![1, 1]), 5), (Index::from(vec![1, 1]), 9)]
        );
        assert_eq!(layout3.input_deps, vec![(Index::from(vec![0, 0]), 8)]);
    }
}
