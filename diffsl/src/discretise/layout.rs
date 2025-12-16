use anyhow::{anyhow, Result};
use itertools::Itertools;
use ndarray::s;
use std::{
    cmp::min,
    convert::AsRef,
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::Arc,
};

use super::{broadcast_shapes, shape::Shape, tensor::Index, TensorBlock};

#[derive(Debug, Clone, PartialEq)]
pub enum LayoutKind {
    Dense,
    Diagonal,
    Sparse,
}

// a sparsity pattern for a multidimensional tensor. A tensor can be sparse, diagonal or dense, as given by the kind field.
// A tensor can also have n_dense_axes axes which are dense, these are the last n_dense_axes axes of the tensor. So for example,
// you could have a 2D sparse tensor with 1 dense axis, combining to give a tensor of rank 3 (i.e. the shape is length 3).
// indices are kept in row major order, so the last index is the fastest changing index.
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    indices: Vec<Index>,
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

        // if layout is dense just remove the last axis
        if self.is_dense() {
            return Ok(Layout::dense(new_shape));
        }

        // if the layout is diagonal and there are no dense axes, then remove the last axis from the shape
        // and the second last axis becomes dense. If we have rank <= 2 then the resultant layout is dense
        if self.is_diagonal() {
            if self.n_dense_axes == 0 && rank > 2 {
                return Ok(Layout {
                    indices: Vec::new(),
                    shape: new_shape,
                    kind: LayoutKind::Diagonal,
                    n_dense_axes: self.n_dense_axes,
                });
            } else if self.n_dense_axes == 0 && rank <= 2 {
                return Ok(Layout::dense(new_shape));
            } else if self.n_dense_axes > 0 {
                return Ok(Layout {
                    indices: Vec::new(),
                    shape: new_shape,
                    kind: LayoutKind::Diagonal,
                    n_dense_axes: self.n_dense_axes - 1,
                });
            }
        }

        // must be sparse
        // if there are no dense axes, then remove the last axis from each index
        if self.n_dense_axes == 0 {
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
                Ok(Layout {
                    indices: new_indices,
                    shape: new_shape,
                    kind: LayoutKind::Dense,
                    n_dense_axes: self.n_dense_axes,
                })
            } else {
                Ok(Layout {
                    indices: new_indices,
                    shape: new_shape,
                    kind: LayoutKind::Sparse,
                    n_dense_axes: self.n_dense_axes,
                })
            }
        } else {
            // there are dense axes, so just remove it
            Ok(Layout {
                indices: Vec::new(),
                shape: new_shape,
                kind: LayoutKind::Sparse,
                n_dense_axes: self.n_dense_axes - 1,
            })
        }
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

        Ok(Layout {
            indices: new_indices,
            shape: new_shape,
            kind: self.kind.clone(),
            n_dense_axes,
        })
    }

    // create a new layout by broadcasting a list of layouts
    // typically different types of layouts cannot be broadcast together, but for multiplies we can broadcast diagonal and dense layouts
    // and sparse and dense layouts
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

        if all_dense {
            return Ok(Layout::dense(shape));
        }

        let any_sparse = layouts.iter().any(|x| x.is_sparse());
        let any_diagonal = layouts.iter().any(|x| x.is_diagonal());
        let all_sparse_and_dense = layouts.iter().all(|x| x.is_sparse() || x.is_dense());
        let all_sparse = layouts.iter().all(|x| x.is_sparse());
        let all_diagonal_and_dense = layouts.iter().all(|x| x.is_diagonal() || x.is_dense());
        let all_diagonal = layouts.iter().all(|x| x.is_diagonal());

        // check for violations of sparse/dense/diagonal rules
        let is_multiply_or_divide = if let Some(op) = op {
            op == '*' || op == '/'
        } else {
            false
        };
        if is_multiply_or_divide {
            if any_diagonal && !all_diagonal_and_dense {
                return Err(anyhow!(
                    "cannot broadcast diagonal and non-dense layouts with multiply"
                ));
            }
            if any_sparse && !all_sparse_and_dense {
                return Err(anyhow!(
                    "cannot broadcast sparse and non-dense layouts with multiply"
                ));
            }
        } else if any_diagonal && !all_diagonal {
            return Err(anyhow!("cannot broadcast diagonal and non-diagonal layouts, except for multiply. Layouts are [{}]", layouts.iter().map(|x| format!("{x}")).join(", ")));
        } else if any_sparse && !all_sparse {
            return Err(anyhow!("cannot broadcast sparse and non-sparse layouts, except for multiply. Layouts are [{}]", layouts.iter().map(|x| format!("{x}")).join(", ")));
        }

        let mut n_dense_axes = None;
        for layout in layouts.iter() {
            if any_diagonal && layout.is_diagonal() || any_sparse && layout.is_sparse() {
                #[allow(clippy::unnecessary_unwrap)]
                if n_dense_axes.is_none() {
                    n_dense_axes = Some(layout.n_dense_axes);
                } else if layout.n_dense_axes != n_dense_axes.unwrap() {
                    return Err(anyhow!(
                        "cannot broadcast layouts with different numbers of dense axes"
                    ));
                }
            }
        }
        let n_dense_axes = n_dense_axes.unwrap();

        // if there are any diagonal layouts then the result is diagonal, all the layouts must be diagonal and have the same number of dense axes
        if any_diagonal {
            return Ok(Layout {
                indices: Vec::new(),
                shape,
                kind: LayoutKind::Diagonal,
                n_dense_axes,
            });
        }

        // if there are any sparse layouts then the result is sparse,
        // and the indicies of all sparse layouts must be identical and have the same number of dense axis.
        // must be sparse and maybe dense
        //
        let mut ret = layouts.pop().unwrap().broadcast_to_shape(&shape);
        for _i in 0..layouts.len() {
            let layout = layouts.pop().unwrap().broadcast_to_shape(&shape);
            if is_multiply_or_divide {
                ret.intersect_inplace(layout)?;
            } else {
                ret.union_inplace(layout)?;
            }
        }

        // if now dense then convert to dense layout
        if ret.indices.len() == ret.shape.product() {
            return Ok(Layout {
                indices: Vec::new(),
                n_dense_axes: ret.shape.len(),
                shape,
                kind: LayoutKind::Dense,
            });
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
    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn new_empty(rank: usize) -> Self {
        Layout {
            indices: vec![],
            shape: Shape::zeros(rank),
            kind: LayoutKind::Dense,
            n_dense_axes: rank,
        }
    }

    pub fn new_scalar() -> Self {
        Layout {
            indices: vec![],
            shape: Shape::zeros(0),
            kind: LayoutKind::Dense,
            n_dense_axes: 0,
        }
    }

    pub fn new_dense(shape: Shape) -> Self {
        let n_dense_axes = shape.len();
        Layout {
            indices: vec![],
            shape,
            kind: LayoutKind::Dense,
            n_dense_axes,
        }
    }

    pub fn new_diagonal(shape: Shape) -> Self {
        Layout {
            indices: vec![],
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

        // if all layouts are dense and contiguous then calculate the new shape and return
        if layouts.iter().all(|x| x.is_dense()) && is_contiguous_on_first_axis {
            // check that this shape can be broadcast into max_shape
            for layout in layouts.iter() {
                for i in 1..std::cmp::min(layout.rank(), rank) {
                    if layout.shape[i] != 1 && layout.shape[i] != max_shape[i] {
                        return Err(anyhow!("cannot concatenate layouts that cannot be broadcast into each other (got shapes {:?})", layouts.iter().map(|x| x.shape()).collect::<Vec<_>>()));
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
            return Ok(Layout::dense(new_shape));
        }

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
            return Ok(Self {
                shape: new_shape,
                indices: Vec::new(),
                kind: LayoutKind::Diagonal,
                n_dense_axes,
            });
        }

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
            shape: max_extent,
            kind: LayoutKind::Sparse,
            n_dense_axes,
        };
        for (layout, start) in std::iter::zip(layouts, starts) {
            // convert to sparse
            new_layout
                .indices
                .extend(layout.indices().map(|x| x + start));
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

        Ok(new_layout)
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
            shape,
            kind: LayoutKind::Dense,
            n_dense_axes,
        }
    }
    pub fn diagonal(shape: Shape) -> Self {
        Self {
            indices: Vec::new(),
            shape,
            kind: LayoutKind::Diagonal,
            n_dense_axes: 0,
        }
    }
    pub fn sparse(indices: Vec<Index>, shape: Shape) -> Self {
        Self {
            indices,
            shape,
            kind: LayoutKind::Sparse,
            n_dense_axes: 0,
        }
    }

    pub fn nnz(&self) -> usize {
        let n_dense: usize = self
            .shape
            .slice(s![self.rank() - self.n_dense_axes..])
            .iter()
            .product();
        if self.is_dense() {
            self.shape.iter().product()
        } else if self.is_diagonal() {
            n_dense
                * (if self.shape.is_empty() {
                    0
                } else {
                    self.shape[0]
                })
        } else {
            n_dense * self.indices.len()
        }
    }

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
    pub fn to_binary_data_layout(&self, other: &Layout) -> Vec<i32> {
        if self == other {
            return vec![];
        }
        if self.is_dense() || self.is_diagonal() {
            return vec![];
        }
        let mut data_layout = vec![];
        for index in other.indices() {
            let nnz_index = self.find_nnz_index(&index).map(|i| i as i32).unwrap_or(-1);
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
            assert_eq!(shape[i], 1);
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

        // if sparse, we need to adjust the indices due to broadcasting
        let mut indices = self.indices.clone();
        if self.is_sparse() {
            for i in 0..self.rank() - self.n_dense_axes {
                if self.shape[i] != shape[i] && self.shape[i] == 1 {
                    let mut new_broadcast_indices = Vec::new();
                    for index in indices.iter() {
                        for j in 0..shape[i] {
                            let mut new_bi = index.clone();
                            new_bi[i] = i64::try_from(j).unwrap();
                            new_broadcast_indices.push(new_bi);
                        }
                    }
                    indices = new_broadcast_indices;
                }
            }
        }
        if self.rank() == shape.len() {
            Self {
                indices,
                shape: shape.clone(),
                kind: self.kind.clone(),
                n_dense_axes: self.n_dense_axes,
            }
        } else if self.rank() < shape.len() {
            let new_ranks = shape.len() - self.rank();
            let n_dense_axes = self.n_dense_axes + new_ranks;
            Self {
                indices: self.indices.clone(),
                shape: shape.clone(),
                kind: self.kind.clone(),
                n_dense_axes,
            }
        } else if self.rank() <= shape.len() && self.rank() - shape.len() <= self.n_dense_axes {
            // must be reducing the rank by a number of dense axes
            let n_dense_axes = self.n_dense_axes - (self.rank() - shape.len());
            Self {
                indices,
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
    pub fn find_nnz_index(&self, index: &Index) -> Option<usize> {
        assert!(index.len() == self.rank());
        let index = {
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
        match self.kind {
            LayoutKind::Sparse => self.indices.iter().position(|x| x == index),
            LayoutKind::Dense => Some(Self::ravel_index(&index, self.shape())),
            LayoutKind::Diagonal => {
                if index.iter().all(|&x| x == index[0])
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

    /// only usable for sparse layouts, adjusts n_dense_axes by expanding the indices
    /// can only reduce n_dense_axes
    fn remove_dense_axes(&mut self, new_n_dense_axes: usize) {
        assert!(self.is_sparse());
        assert!(new_n_dense_axes <= self.n_dense_axes);
        if self.n_dense_axes == new_n_dense_axes {
            return;
        }
        let rank = self.rank();
        let removed_axes = self.n_dense_axes - new_n_dense_axes;
        let mut new_indices = Vec::new();
        for index in self.indices.iter() {
            let dense_shape = self.shape.slice(s![rank - removed_axes..]).to_owned();
            let n_dense: usize = dense_shape.iter().product();
            for i in 0..n_dense {
                let mut new_index = Index::zeros(rank);
                for j in 0..rank - removed_axes {
                    new_index[j] = index[j];
                }
                let dense_index = Self::unravel_index(i, &dense_shape);
                for j in 0..removed_axes {
                    new_index[rank - removed_axes + j] = dense_index[j];
                }
                new_indices.push(new_index);
            }
        }
        self.indices = new_indices;
        self.n_dense_axes = 0;
    }

    /// both self and other should be sparse layouts with the same shape
    /// the result is the union of the two layouts
    /// Note: one of teh layouts could have a different number of dense axes, in which case
    /// the dense axes are removed from the layout with more dense axes
    pub fn union_inplace(&mut self, mut other: Layout) -> Result<()> {
        if !self.is_sparse() || !other.is_sparse() {
            return Err(anyhow!("can only union sparse layouts"));
        }
        if self.shape != other.shape {
            return Err(anyhow!(
                "can only union layouts with the same shape and number of dense axes"
            ));
        }
        if self.n_dense_axes > other.n_dense_axes {
            self.remove_dense_axes(other.n_dense_axes);
        } else if other.n_dense_axes > self.n_dense_axes {
            other.remove_dense_axes(self.n_dense_axes);
        }
        self.indices.extend(other.indices.iter().cloned());
        self.indices.sort_by(Self::cmp_index);
        self.indices.dedup();
        Ok(())
    }

    /// self is a sparse layout, other is either a sparse or dense layout with the same shape and n_dense_axes
    /// the result is the intersection of the two layouts
    pub fn intersect_inplace(&mut self, mut other: Layout) -> Result<()> {
        if !self.is_sparse() {
            return Err(anyhow!("can only intersect sparse layouts"));
        }
        if !other.is_sparse() && !other.is_dense() {
            return Err(anyhow!("can only intersect with sparse or dense layouts"));
        }
        if self.shape != other.shape {
            return Err(anyhow!("can only intersect layouts with the same shape"));
        }
        if other.is_dense() {
            return Ok(());
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
        Ok(())
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
        let broadcasted2 = Layout::broadcast(vec![layout1.clone(), layout2.clone()], Some('+'));
        assert!(broadcasted2.is_err());
        let broadcasted3 = Layout::broadcast(vec![layout1.clone(), layout3.clone()], Some('*'))?;
        assert!(broadcasted3.is_sparse());
        let broadcasted4 = Layout::broadcast(vec![layout1.clone(), layout3.clone()], Some('+'));
        assert!(broadcasted4.is_err());
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
        let data_layout = layout1.to_binary_data_layout(&layout1_plus_layout2);
        assert_eq!(data_layout, vec![0, -1]);
        let data_layout = layout2.to_binary_data_layout(&layout1_plus_layout2);
        assert_eq!(data_layout, vec![]);
        Ok(())
    }

    #[test]
    fn test_union_sparse_layouts() -> Result<()> {
        let mut layout1 = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        layout1.union_inplace(layout2)?;
        assert_eq!(layout1.indices.len(), 3);

        let mut layout1 = Layout::sparse(vec![Index::from(vec![1])], Shape::from(vec![2, 2]));
        layout1.n_dense_axes = 1;
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        layout1.union_inplace(layout2)?;
        assert_eq!(layout1.indices.len(), 2);
        assert_eq!(layout1.indices[0], Index::from(vec![1, 0]));
        assert_eq!(layout1.indices[1], Index::from(vec![1, 1]));
        Ok(())
    }

    #[test]
    fn test_intersect_sparse_layouts() -> Result<()> {
        let mut layout1 = Layout::sparse(
            vec![Index::from(vec![0, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        let layout2 = Layout::sparse(
            vec![Index::from(vec![1, 0]), Index::from(vec![1, 1])],
            Shape::from(vec![2, 2]),
        );
        layout1.intersect_inplace(layout2)?;
        assert_eq!(layout1.indices.len(), 1);
        assert_eq!(layout1.indices[0], Index::from(vec![1, 1]));

        let mut layout1 = Layout::sparse(vec![Index::from(vec![1])], Shape::from(vec![2, 2]));
        layout1.n_dense_axes = 1;
        let layout2 = Layout::sparse(vec![Index::from(vec![1, 0])], Shape::from(vec![2, 2]));
        layout1.intersect_inplace(layout2)?;
        assert_eq!(layout1.indices.len(), 1);
        assert_eq!(layout1.indices[0], Index::from(vec![1, 0]));
        Ok(())
    }
}
