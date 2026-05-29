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

pub fn read_sparse_tensor(path: &str, shape: &Shape) -> Result<Layout> {
    match SparseImportFormat::from_path(path)? {
        SparseImportFormat::Frostt => read_frostt(path, shape),
    }
}

fn read_frostt(path: &str, shape: &Shape) -> Result<Layout> {
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

    Ok(Layout::from_sparse_values(entries, shape.clone()))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn write_temp_tns(name: &str, contents: &str) -> String {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("diffsl_{name}_{unique}.tns"));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn frostt_import_builds_dense_layout_when_all_entries_are_present() {
        let path = write_temp_tns(
            "frostt_dense_layout",
            "
            2 2 4.0
            1 1 1.0
            2 1 3.0
            1 2 2.0
            ",
        );
        let layout = read_sparse_tensor(&path, &Shape::from_vec(vec![2, 2])).unwrap();

        assert!(layout.is_dense());
        assert_eq!(
            layout.indices().map(|i| i.to_string()).collect::<Vec<_>>(),
            vec!["[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]"]
        );
        assert_eq!(layout.values().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn frostt_import_builds_diagonal_layout_when_all_diagonal_entries_are_present() {
        let path = write_temp_tns(
            "frostt_diagonal_layout",
            "
            3 3 9.0
            1 1 1.0
            2 2 4.0
            ",
        );
        let layout = read_sparse_tensor(&path, &Shape::from_vec(vec![3, 3])).unwrap();

        assert!(layout.is_diagonal());
        assert_eq!(
            layout.indices().map(|i| i.to_string()).collect::<Vec<_>>(),
            vec!["[0, 0]", "[1, 1]", "[2, 2]"]
        );
        assert_eq!(layout.values().unwrap(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn frostt_import_builds_sparse_layout_for_partial_non_diagonal_entries() {
        let path = write_temp_tns(
            "frostt_sparse_layout",
            "
            2 3 5.0
            1 1 2.0
            ",
        );
        let layout = read_sparse_tensor(&path, &Shape::from_vec(vec![3, 3])).unwrap();

        assert!(layout.is_sparse());
        assert_eq!(
            layout.indices().map(|i| i.to_string()).collect::<Vec<_>>(),
            vec!["[0, 0]", "[1, 2]"]
        );
        assert_eq!(layout.values().unwrap(), &[2.0, 5.0]);
    }
}
