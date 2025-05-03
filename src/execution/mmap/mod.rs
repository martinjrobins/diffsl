#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
pub(crate) use wasm::{Mmap, MmapMut, MmapOptions};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use mmap_rs::{Mmap, MmapMut, MmapOptions};

use anyhow::{anyhow, Result};

pub(crate) enum MappedSection {
    Mutable(MmapMut),
    Immutable(Mmap),
}

impl MappedSection {
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            MappedSection::Mutable(map) => map.as_ptr(),
            MappedSection::Immutable(map) => map.as_ptr(),
        }
    }
    pub fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        match self {
            MappedSection::Mutable(map) => Some(map.as_mut_ptr()),
            MappedSection::Immutable(_map) => None,
        }
    }
    pub fn make_read_only(self) -> Result<Self> {
        match self {
            MappedSection::Mutable(map) => Ok(MappedSection::Immutable(
                map.make_read_only()
                    .map_err(|e| anyhow!("Failed to make section read-only: {:?}", e))?,
            )),
            MappedSection::Immutable(_) => Ok(self),
        }
    }
    pub fn make_exec(self) -> Result<Self> {
        match self {
            MappedSection::Mutable(_map) => Err(anyhow!("Cannot make mutable section executable")),
            MappedSection::Immutable(map) => {
                Ok(MappedSection::Immutable(map.make_exec().map_err(|e| {
                    anyhow!("Failed to make section executable: {:?}", e)
                })?))
            }
        }
    }
}
