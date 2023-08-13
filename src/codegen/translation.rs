use std::fmt;



#[derive(Debug, Clone, PartialEq)]
pub enum TranslationFrom {
    // contraction over a dense expression. contract by the last `contract_by` axes, which are of len `contract_len`
    DenseContraction{ contract_by: usize, contract_len: usize },

    // contraction over a diagonal expression. contract by the last `contract_by` axes, which are of len `contract_len`
    DiagonalContraction{ contract_by: usize },

    // contraction over a sparse expression, each contraction starts at the given start index and ends at the given end index
    SparseContraction{ contract_by: usize, contract_start_indices: Vec<usize>, contract_end_indices: Vec<usize> },

    // each nz of the sparse expression is summed into a corresponding nz of the target tensor (given by the TranslationTo)
    // used for all types of expressions
    ElementWise,

    // broadcast each expr nz to the subsequent `broadcast_len` elements in the tensor
    // corresponding to the last broadcast_by axes of the tensor
    // used for all types of expressions
    Broadcast{ broadcast_by: usize, broadcast_len: usize },

}

impl fmt::Display for TranslationFrom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslationFrom::DenseContraction{ contract_by, contract_len } => write!(f, "DenseContraction({}, {})", contract_by, contract_len),
            TranslationFrom::DiagonalContraction{ contract_by } => write!(f, "DiagonalContraction({})", contract_by),
            TranslationFrom::SparseContraction{ contract_by, contract_start_indices, contract_end_indices } => write!(f, "SparseContraction({}, {:?}, {:?})", contract_by, contract_start_indices, contract_end_indices),
            TranslationFrom::ElementWise => write!(f, "ElementWise"),
            TranslationFrom::Broadcast{ broadcast_by, broadcast_len } => write!(f, "Broadcast({}, {})", broadcast_by, broadcast_len),
        }
    }
}

impl TranslationFrom {
    // traslate from source layout (an expression) via an intermediary target layout (a tensor block)
    fn new(source: &Layout, target: &Layout) -> Self {
        let mut min_rank_for_broadcast = source.rank();
        if source.rank() <= target.rank() {
            for i in (0..source.rank()).rev() {
                if source.shape()[i] != target.shape()[i] {
                    assert!(source.shape()[i] == 1);
                    min_rank_for_broadcast = i + 1;
                    break;
                }
            }
        }
        let broadcast_by = if target.rank() >= min_rank_for_broadcast { target.rank() - min_rank_for_broadcast } else { 0 };
        let contract_by = if source.rank() >= target.rank() { source.rank() - target.rank() } else { 0 };
        let is_broadcast = broadcast_by > 0;
        let is_contraction = source.rank() > target.rank();


        if source.is_dense() && is_contraction {
            Self::DenseContraction{ contract_by, contract_len: source.shape().slice(s![contract_by..]).iter().product() }
        } else if source.is_diagonal() && is_contraction {
            Self::DiagonalContraction{ contract_by }
        } else if source.is_sparse() && is_contraction {
            let mut contract_start_indices = vec![0];
            let mut contract_end_indices = Vec::new();
            let monitor_axis = source.rank() - contract_by - 1;
            let indices: Vec<Index> = source.indices().collect();
            let mut current_monitor_axis_value = indices[0][monitor_axis];
            // the indices are held in row major order, so the last index is the fastest changing index
            for i in 1..indices.len() {
                let index = &indices[i];
                let monitor_axis_value = index[monitor_axis];
                if monitor_axis_value != current_monitor_axis_value {
                    contract_start_indices.push(i);
                    contract_end_indices.push(i);
                    current_monitor_axis_value = monitor_axis_value;
                }
            }
            contract_end_indices.push(indices.len());
            assert!(contract_start_indices.len() == contract_end_indices.len());
            assert!(contract_start_indices.len() == target.nnz());
            Self::SparseContraction{ contract_by, contract_start_indices, contract_end_indices }
        } else if is_broadcast {
            if target.n_dense_axes >= broadcast_by {
                let broadcast_len = target.shape().slice(s![min_rank_for_broadcast..]).iter().product();
                Self::Broadcast{ broadcast_by, broadcast_len }
            } else if target.is_diagonal() {
                let broadcast_len = target.shape()[0];
                Self::Broadcast{ broadcast_by, broadcast_len }
            } else {
                panic!("invalid broadcast")
            }
        } else {
            Self::ElementWise
        }
    }
    fn nnz_after_translate(&self, layout: &Layout) -> usize {
        match self {
            TranslationFrom::DenseContraction{ contract_by: _, contract_len } => layout.nnz() / contract_len,
            TranslationFrom::DiagonalContraction{ contract_by: _} => layout.nnz(),
            TranslationFrom::SparseContraction{ contract_by: _, contract_start_indices, contract_end_indices: _ } => contract_start_indices.len(),
            TranslationFrom::ElementWise => layout.nnz(),
            TranslationFrom::Broadcast{ broadcast_by: _, broadcast_len } => layout.nnz() * broadcast_len,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranslationTo {
    // indices in the target tensor nz array are contiguous and start/end at the given indices, end is exclusive
    Contiguous{ start: usize, end: usize },

    // indices in the target tensor nz array are given by the indices in the given vector
    Sparse{ indices: Vec<usize> },
}

impl fmt::Display for TranslationTo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslationTo::Contiguous{ start, end } => write!(f, "Contiguous({}, {})", start, end),
            TranslationTo::Sparse{ indices } => write!(f, "Sparse{:?}", indices),
        }
    }
}

impl TranslationTo {
    // start is the index of the first element in the target tensor
    // sourse is the layout of the target tensor block
    // target is the layout of the target tensor
    fn new(start: &Index, source: &Layout, target: &Layout) -> Self {
        if target.is_dense() || target.is_diagonal() {
            let start = target.find_nnz_index(start).unwrap();
            let end = start + source.nnz();
            TranslationTo::Contiguous{ start, end }
        } else if target.is_sparse() {
            let indices: Vec<usize> = source.indices().map(|index| {
                target.find_nnz_index(&(index + start)).unwrap()
            }).collect();
            // check if the indices are contiguous
            let contiguous = indices.windows(2).all(|w| w[1] == w[0] + 1);
            if contiguous {
                let start = indices[0];
                let end = indices[indices.len() - 1] + 1;
                TranslationTo::Contiguous{ start, end }
            } else {
                TranslationTo::Sparse{ indices }
            }
        } else {
            panic!("invalid target layout")
        }
    }
    fn nnz_after_translate(&self) -> usize {
        match self {
            TranslationTo::Contiguous{ start, end } => end - start,
            TranslationTo::Sparse{ indices } => indices.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Translation {
    pub source: TranslationFrom,
    pub target: TranslationTo,
}

impl fmt::Display for Translation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Translation({}, {})", self.source, self.target)
    }
}

impl Translation {
    pub fn new(source: &RcLayout, via: &RcLayout, target_start: &Index, target: &RcLayout) -> Self {
        let source_layout = source.borrow();
        let target_layout = target.borrow();
        let via_layout = via.borrow();
        let from = TranslationFrom::new(source_layout, via_layout);
        let to = TranslationTo::new(target_start, via_layout, target_layout);
        assert_eq!(from.nnz_after_translate(source_layout), to.nnz_after_translate());
        Self { source: from, target: to}
    }
    pub fn to_data_layout(&self) -> Vec<i32> {
        let mut ret = Vec::new();
        if let TranslationFrom::SparseContraction { contract_by: _, contract_start_indices, contract_end_indices } = &self.source {
            ret.extend(contract_start_indices.iter().zip(contract_end_indices.iter()).flat_map(|(start, end)| vec![i32::try_from(*start).unwrap(), i32::try_from(*end).unwrap()]));
        }
        if let TranslationTo::Sparse { indices } = &self.target {
            ret.extend(indices.iter().map(|i| *i as i32));
        }
        ret
    }
    pub fn get_from_index_in_data_layout(&self) -> usize {
        0
    }
    pub fn get_to_index_in_data_layout(&self) -> usize {
        if let TranslationFrom::SparseContraction { contract_by: _, contract_start_indices, contract_end_indices: _ } = &self.source {
            contract_start_indices.len() * 2
        } else {
            0
        }
    }
}
