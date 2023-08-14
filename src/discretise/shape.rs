use ndarray::Array1;



pub type Shape = Array1<usize>;

pub fn broadcast_shapes(shapes: &[&Shape]) -> Option<Shape> {
    if shapes.is_empty() {
        return None;
    }
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap();
    let mut shape = Shape::zeros(max_rank);
    for i in (0..max_rank).rev() {
        let mdim = shapes.iter().map(|s| *s.get(i).unwrap_or(&1)).max().unwrap();
        let compatible = shapes.iter().all(|s| if let Some(x) = s.get(i) { *x == mdim || *x == 1 } else { true });
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