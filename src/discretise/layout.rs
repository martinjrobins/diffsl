use anyhow::{anyhow, Result};
use itertools::Itertools;
use ndarray::s;
use std::{convert::AsRef, fmt, hash::Hash, hash::Hasher, ops::Deref, rc::Rc};

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
                let mut new_i = i.slice(s![..new_rank]).to_owned();
                for (ai, &p) in permutation.iter().enumerate() {
                    new_i[ai] = i[p];
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
    pub fn broadcast(mut layouts: Vec<Layout>, is_multiply: bool) -> Result<Layout> {
        // the shapes of the layouts must be broadcastable
        let shapes = layouts.iter().map(|x| &x.shape).collect::<Vec<_>>();
        let shape = match broadcast_shapes(&shapes[..]) {
            Some(x) => x,
            None => {
                let shapes_str = shapes
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(anyhow!("cannot broadcast shapes: {}", shapes_str));
            }
        };

        let all_dense = layouts.iter().all(|x| x.is_dense());

        if all_dense {
            return Ok(Layout::dense(shape));
        }

        let all_sparse = layouts.iter().all(|x| x.is_sparse());
        let all_diagonal = layouts.iter().all(|x| x.is_diagonal());
        let any_sparse = layouts.iter().any(|x| x.is_sparse());
        let any_diagonal = layouts.iter().any(|x| x.is_diagonal());
        let all_sparse_and_dense = layouts.iter().all(|x| x.is_sparse() || x.is_dense());
        let all_diagonal_and_dense = layouts.iter().all(|x| x.is_diagonal() || x.is_dense());

        // check for violations of sparse/dense/diagonal rules
        let n_dense_axes = if is_multiply {
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
            let mut n_dense_axes = None;
            for layout in layouts.iter() {
                if any_diagonal && layout.is_diagonal() || any_sparse && layout.is_sparse() {
                    if n_dense_axes.is_none() {
                        n_dense_axes = Some(layout.n_dense_axes);
                    } else if layout.n_dense_axes != n_dense_axes.unwrap() {
                        return Err(anyhow!(
                            "cannot broadcast layouts with different numbers of dense axes"
                        ));
                    }
                }
            }
            n_dense_axes.unwrap()
        } else {
            if any_diagonal && !all_diagonal {
                return Err(anyhow!("cannot broadcast diagonal and non-diagonal layouts, except for multiply. Layouts are [{}]", layouts.iter().map(|x| format!("{}", x)).join(", ")));
            }
            if any_sparse && !all_sparse {
                return Err(anyhow!("cannot broadcast sparse and non-sparse layouts, except for multiply. Layouts are [{}]", layouts.iter().map(|x| format!("{}", x)).join(", ")));
            }
            if layouts
                .iter()
                .any(|x| x.n_dense_axes != layouts[0].n_dense_axes)
            {
                return Err(anyhow!("cannot broadcast diagonal layouts with different numbers of dense axes. Layouts are [{}]", layouts.iter().map(|x| format!("{}", x)).join(", ")));
            }
            layouts[0].n_dense_axes
        };

        // if there are any diagonal layouts then the result is diagonal, all the layouts must be diagonal and have the same number of dense axes
        if any_diagonal {
            return Ok(Layout {
                indices: Vec::new(),
                shape,
                kind: LayoutKind::Diagonal,
                n_dense_axes,
            });
        }

        // if there are any sparse layouts then the result is sparse, and the indicies of all sparse layouts must be identical and have the same number of dense axis.
        // must be sparse and maybe dense
        let mut indices = None;
        for _i in 0..layouts.len() {
            let layout = layouts.pop().unwrap();
            if layout.is_sparse() {
                if indices.is_none() {
                    indices = Some(layout.indices);
                } else if layout.indices.len() != indices.as_ref().unwrap().len()
                    || layout
                        .indices
                        .iter()
                        .zip(indices.as_ref().unwrap().iter())
                        .any(|(x, y)| x != y)
                {
                    return Err(anyhow!(
                        "cannot broadcast layouts with different sparsity patterns"
                    ));
                }
            }
        }
        Ok(Layout {
            indices: indices.unwrap(),
            shape,
            kind: LayoutKind::Sparse,
            n_dense_axes,
        })
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

    pub fn to_data_layout(&self) -> Vec<i32> {
        let mut data_layout = vec![];
        if self.is_sparse() {
            for index in self.indices() {
                data_layout.extend(index.iter().map(|&x| x as i32));
            }
        }
        data_layout
    }

    pub fn to_rank(&self, rank: usize) -> Option<Self> {
        if self.rank() == rank {
            Some(self.clone())
        } else if self.rank() < rank {
            // must be increasing the rank
            let new_ranks = rank - self.rank();
            let shape = Shape::from_iter(
                self.shape
                    .iter()
                    .cloned()
                    .chain(std::iter::repeat(1).take(new_ranks)),
            );
            let n_dense_axes = self.n_dense_axes + new_ranks;
            Some(Self {
                indices: self.indices.clone(),
                shape,
                kind: self.kind.clone(),
                n_dense_axes,
            })
        } else if self.min_rank() <= rank && self.rank() - rank <= self.n_dense_axes {
            // must be reducing the rank by a number of dense axes
            let shape = self.shape.slice(s![..rank]).to_owned();
            let n_dense_axes = self.n_dense_axes - (self.rank() - rank);
            Some(Self {
                indices: self.indices.clone(),
                shape,
                kind: self.kind.clone(),
                n_dense_axes,
            })
        } else {
            // invalid
            None
        }
    }

    // returns the index in the nnz array corresponding to the given index
    pub fn find_nnz_index(&self, index: &Index) -> Option<usize> {
        match self.kind {
            LayoutKind::Sparse => self.indices.iter().position(|x| x == index),
            LayoutKind::Dense => {
                let valid_index = ndarray::Zip::from(index)
                    .and(self.shape())
                    .all(|&a, &b| a < b.try_into().unwrap());
                if valid_index {
                    Some(Self::ravel_index(index, self.shape()))
                } else {
                    None
                }
            }
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
}

// RcLayout is a wrapper for Rc<Layout> that implements Hash and PartialEq based on ptr equality
#[derive(Debug)]
pub struct RcLayout(Rc<Layout>);
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
impl AsRef<Layout> for RcLayout {
    fn as_ref(&self) -> &Layout {
        &self.0
    }
}
impl RcLayout {
    pub fn new(layout: Layout) -> Self {
        Self(Rc::new(layout))
    }
}
