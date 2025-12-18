use std::collections::HashMap;

use crate::discretise::{ArcLayout, DiscreteModel, Layout, Tensor, TensorBlock};

use super::Translation;

// there are three different layouts:
// 1. the data layout is a mapping from tensors to the index of the first element in the data or constants array.
//    Each tensor in the data or constants layout is a contiguous array of nnz elements
// 2. the layout layout is a mapping from Layout to the index of the first element in the indices array.
//    Only sparse layouts are stored, and each sparse layout is a contiguous array of nnz*rank elements
// 3. the translation layout is a mapping from layout from-to pairs to the index of the first element in the indices array.
//    Each contraction pair is an array of nnz-from elements, each representing the indices of the "to" tensor that will be summed into.
// We also store a mapping from tensor names to their layout, so that we can easily look up the layout of a tensor
#[derive(Debug)]
pub struct DataLayout {
    is_constant_map: HashMap<String, bool>,
    data_index_map: HashMap<String, usize>,
    data_length_map: HashMap<String, usize>,
    layout_index_map: HashMap<ArcLayout, usize>,
    binary_layout_index_map: HashMap<(ArcLayout, ArcLayout, Vec<usize>), usize>,
    translate_index_map: HashMap<(ArcLayout, ArcLayout), usize>,
    data: Vec<f64>,
    constants: Vec<f64>,
    indices: Vec<i32>,
    layout_map: HashMap<String, ArcLayout>,
}

impl DataLayout {
    pub fn new(model: &DiscreteModel) -> Self {
        let mut is_constant_map = HashMap::new();
        let mut data_index_map = HashMap::new();
        let mut data_length_map = HashMap::new();
        let mut layout_index_map = HashMap::new();
        let mut translate_index_map = HashMap::new();
        let mut data = Vec::new();
        let mut constants = Vec::new();
        let mut indices = Vec::new();
        let mut layout_map = HashMap::new();
        let mut binary_layout_index_map = HashMap::new();

        // add layout info for "t"
        let t_layout = ArcLayout::new(Layout::new_scalar());
        layout_map.insert("t".to_string(), t_layout);

        let mut add_tensor = |tensor: &Tensor, in_data: bool, in_constants: bool| {
            // insert the data (non-zeros) for each tensor
            layout_map.insert(tensor.name().to_string(), tensor.layout_ptr().clone());
            if in_data {
                data_index_map.insert(tensor.name().to_string(), data.len());
                data_length_map.insert(tensor.name().to_string(), tensor.nnz());
                data.extend(vec![0.0; tensor.nnz()]);
                is_constant_map.insert(tensor.name().to_string(), false);
            } else if in_constants {
                data_index_map.insert(tensor.name().to_string(), constants.len());
                data_length_map.insert(tensor.name().to_string(), tensor.nnz());
                constants.extend(vec![0.0; tensor.nnz()]);
            }
            is_constant_map.insert(tensor.name().to_string(), in_constants);

            // add the translation info for each block-tensor pair
            for blk in tensor.elmts() {
                // need layouts and is_constant of all named tensor blocks
                if let Some(name) = blk.name() {
                    layout_map.insert(name.to_string(), blk.layout().clone());
                    is_constant_map.insert(name.to_string(), in_constants);
                }

                // insert the layout info for each tensor expression
                if !layout_index_map.contains_key(blk.expr_layout()) {
                    layout_index_map.insert(blk.expr_layout().clone(), indices.len());
                    indices.extend(blk.expr_layout().to_data_layout());
                }

                // if any tensors in the block expression have a different layout to the block expression
                // then we need to add a binary layout translation
                for (tensor_name, tensor_indices) in blk.expr().get_dependents_with_indices() {
                    let tensor_layout = layout_map.get(tensor_name).unwrap();
                    if tensor_layout != blk.expr_layout() {
                        let permutation = Self::permutation(blk, &tensor_indices, tensor_layout);
                        if !binary_layout_index_map.contains_key(&(
                            tensor_layout.clone(),
                            blk.expr_layout().clone(),
                            permutation.clone(),
                        )) {
                            let blayout = tensor_layout
                                .to_binary_data_layout(blk.expr_layout(), &permutation);
                            binary_layout_index_map.insert(
                                (
                                    tensor_layout.clone(),
                                    blk.expr_layout().clone(),
                                    permutation,
                                ),
                                indices.len(),
                            );
                            indices.extend(blayout);
                        }
                    }
                }

                // and the translation info for each block-tensor pair
                if let std::collections::hash_map::Entry::Vacant(e) =
                    translate_index_map.entry((blk.expr_layout().clone(), blk.layout().clone()))
                {
                    let translation = Translation::new(
                        blk.expr_layout(),
                        blk.layout(),
                        blk.start(),
                        tensor.layout_ptr(),
                    );
                    e.insert(indices.len());
                    indices.extend(translation.to_data_layout());
                }
            }
        };

        model
            .constant_defns()
            .iter()
            .for_each(|c| add_tensor(c, false, true));
        model
            .inputs()
            .iter()
            .for_each(|i| add_tensor(i, true, false));
        model
            .input_dep_defns()
            .iter()
            .for_each(|i| add_tensor(i, true, false));

        model
            .time_dep_defns()
            .iter()
            .for_each(|i| add_tensor(i, true, false));

        add_tensor(model.state(), false, false);
        if let Some(state_dot) = model.state_dot() {
            add_tensor(state_dot, false, false);
        }
        model
            .state_dep_defns()
            .iter()
            .for_each(|i| add_tensor(i, true, false));
        if let Some(lhs) = model.lhs() {
            add_tensor(lhs, false, false);
        }
        add_tensor(model.rhs(), false, false);
        if let Some(out) = model.out() {
            add_tensor(out, false, false);
        }

        // todo: could we just calculate constants now?

        Self {
            is_constant_map,
            data_index_map,
            layout_index_map,
            data,
            indices,
            translate_index_map,
            layout_map,
            data_length_map,
            constants,
            binary_layout_index_map,
        }
    }

    /// construct a permutation from the block expression indices to the tensor indices
    /// in case they are in a different order
    /// if any indices appear in the tensor indices but not in the block indices, we add these
    /// to the end of the permutation (these will be contracted indices)
    /// if any indices appear in the block indices but not in the tensor indices, we
    /// map them to the end (these will be broadcasted indices)
    ///
    /// case 1: no contraction, translate
    /// (i, j) -> (j, i) permutation [1, 0]
    /// case 2: contraction, translate, always contract last index
    /// (i) -> (j, i) permutation [1, 1]
    /// case 3: contraction with tranlation with broadcast
    /// (i) -> (j) permutation [1, 0]
    pub fn permutation(
        blk: &TensorBlock,
        tensor_indices: &[char],
        tensor_layout: &ArcLayout,
    ) -> Vec<usize> {
        let mut permutation = blk
            .indices()
            .iter()
            .map(|idx| {
                tensor_indices
                    .iter()
                    .position(|&c| c == *idx)
                    .unwrap_or(tensor_layout.rank())
            })
            .collect::<Vec<usize>>();
        for (i, index) in tensor_indices.iter().enumerate() {
            if !blk.indices().contains(index) {
                permutation.push(i);
            }
        }
        permutation
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&String, bool)> {
        self.data_index_map
            .keys()
            .map(|name| (name, *self.is_constant_map.get(name).unwrap()))
    }

    // get the layout of a tensor by name
    pub fn get_layout(&self, name: &str) -> Option<&ArcLayout> {
        self.layout_map.get(name)
    }

    pub fn is_constant(&self, name: &str) -> bool {
        *self.is_constant_map.get(name).unwrap()
    }

    // get the index of the data array for the given tensor name
    pub fn get_data_index(&self, name: &str) -> Option<usize> {
        self.data_index_map.get(name).copied()
    }

    pub fn format_data(&self, data: &[f64]) -> String {
        let mut data_index_sorted: Vec<_> = self.data_index_map.iter().collect();
        data_index_sorted.sort_by_key(|(_, index)| **index);
        let mut s = String::new();
        s += "[";
        for (name, index) in data_index_sorted {
            let nnz = self.data_length_map[name];
            s += &format!("{}: {:?}, ", name, &data[*index..*index + nnz]);
        }
        s += "]";
        s
    }

    pub fn get_tensor_data(&self, name: &str) -> Option<&[f64]> {
        let index = self.get_data_index(name)?;
        let nnz = self.get_data_length(name)?;
        Some(&self.data()[index..index + nnz])
    }
    pub fn get_tensor_data_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        let index = self.get_data_index(name)?;
        let nnz = self.get_data_length(name)?;
        Some(&mut self.data_mut()[index..index + nnz])
    }

    pub fn get_data_length(&self, name: &str) -> Option<usize> {
        self.data_length_map.get(name).copied()
    }

    pub fn get_layout_index(&self, layout: &ArcLayout) -> Option<usize> {
        self.layout_index_map.get(layout).copied()
    }

    pub fn get_binary_layout_index(
        &self,
        from: &ArcLayout,
        to: &ArcLayout,
        permutation: Vec<usize>,
    ) -> Option<usize> {
        self.binary_layout_index_map
            .get(&(from.clone(), to.clone(), permutation))
            .copied()
    }

    pub fn get_translation_index(&self, from: &ArcLayout, to: &ArcLayout) -> Option<usize> {
        self.translate_index_map
            .get(&(from.clone(), to.clone()))
            .copied()
    }

    pub fn data(&self) -> &[f64] {
        self.data.as_ref()
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        self.data.as_mut_slice()
    }

    pub fn constants(&self) -> &[f64] {
        self.constants.as_ref()
    }

    pub fn constants_mut(&mut self) -> &mut [f64] {
        self.constants.as_mut_slice()
    }

    pub fn indices(&self) -> &[i32] {
        self.indices.as_ref()
    }
}
