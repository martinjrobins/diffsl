use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

use super::{Index, Layout, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseImportFormat {
    Frostt,
}

impl SparseImportFormat {
    pub fn from_path(path: &str) -> Result<Self> {
        match Path::new(path).extension().and_then(|ext| ext.to_str()) {
            Some("tns") => Ok(Self::Frostt),
            Some(ext) => Err(anyhow!("unsupported sparse tensor file extension '.{ext}'")),
            None => Err(anyhow!("sparse tensor file path must have an extension")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SparseImportData {
    indices: Vec<Index>,
    values: Vec<f64>,
}

impl SparseImportData {
    pub fn new(indices: Vec<Index>, values: Vec<f64>) -> Self {
        Self { indices, values }
    }

    pub fn indices(&self) -> &[Index] {
        self.indices.as_slice()
    }

    pub fn values(&self) -> &[f64] {
        self.values.as_slice()
    }
}

pub fn read_sparse_tensor(path: &str, shape: &Shape) -> Result<SparseImportData> {
    match SparseImportFormat::from_path(path)? {
        SparseImportFormat::Frostt => read_frostt(path, shape),
    }
}

fn read_frostt(path: &str, shape: &Shape) -> Result<SparseImportData> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read sparse tensor file '{path}'"))?;
    let rank = shape.len();
    let mut entries = Vec::new();
    let mut seen = HashSet::new();

    for (line_index, raw_line) in contents.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != rank + 1 {
            return Err(anyhow!(
                "invalid FROSTT row in '{path}' at line {line_number}: expected {} fields, got {}",
                rank + 1,
                parts.len()
            ));
        }

        let mut index = Vec::with_capacity(rank);
        for axis in 0..rank {
            let coord = parts[axis].parse::<i64>().with_context(|| {
                format!(
                    "invalid FROSTT coordinate '{}' in '{path}' at line {line_number}",
                    parts[axis]
                )
            })?;
            if coord <= 0 {
                return Err(anyhow!(
                    "invalid FROSTT coordinate {coord} in '{path}' at line {line_number}: coordinates are 1-based"
                ));
            }
            let zero_based = coord - 1;
            if zero_based >= i64::try_from(shape[axis]).unwrap() {
                return Err(anyhow!(
                    "FROSTT coordinate {coord} in '{path}' at line {line_number} is outside axis {axis} extent {}",
                    shape[axis]
                ));
            }
            index.push(zero_based);
        }

        if !seen.insert(index.clone()) {
            return Err(anyhow!(
                "duplicate FROSTT coordinate {:?} in '{path}' at line {line_number}",
                index
            ));
        }

        let value = parts[rank].parse::<f64>().with_context(|| {
            format!(
                "invalid FROSTT value '{}' in '{path}' at line {line_number}",
                parts[rank]
            )
        })?;
        entries.push((Index::from_vec(index), value));
    }

    entries.sort_by(|(left, _), (right, _)| Layout::cmp_index(left, right));
    let (indices, values) = entries.into_iter().unzip();
    Ok(SparseImportData::new(indices, values))
}
