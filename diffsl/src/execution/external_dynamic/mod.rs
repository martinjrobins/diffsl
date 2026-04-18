use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use object::{Object, ObjectSection, ObjectSymbol, SymbolKind};

mod loader;

use super::{
    external_interface::{is_external_symbol_name, normalize_symbol_name},
    module::{CodegenModule, CodegenModuleJit},
};
use loader::DynamicLibrary;

pub struct ExternalDynModule<T> {
    path: PathBuf,
    _library: DynamicLibrary,
    symbols: HashMap<String, usize>,
    _marker: PhantomData<T>,
}

impl<T> ExternalDynModule<T> {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let library = DynamicLibrary::open(&path)?;
        let symbols = discover_symbols(&path, &library)?;
        Ok(Self {
            path,
            _library: library,
            symbols,
            _marker: PhantomData,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl<T> CodegenModule for ExternalDynModule<T> where T: Send + Sync + 'static {}

impl<T> CodegenModuleJit for ExternalDynModule<T>
where
    T: Send + Sync + 'static,
{
    fn jit(&mut self) -> Result<HashMap<String, *const u8>> {
        Ok(self
            .symbols
            .iter()
            .map(|(name, addr)| (name.clone(), *addr as *const u8))
            .collect())
    }
}

fn discover_symbols(path: &Path, library: &DynamicLibrary) -> Result<HashMap<String, usize>> {
    let buffer = std::fs::read(path).with_context(|| {
        format!(
            "Failed to read dynamic library for symbol discovery: {}",
            path.display()
        )
    })?;
    let file = object::File::parse(buffer.as_slice()).with_context(|| {
        format!(
            "Failed to parse dynamic library for symbol discovery: {}",
            path.display()
        )
    })?;

    let mut discovered = BTreeMap::new();
    collect_file_symbols(&file, &mut discovered)?;

    if discovered.is_empty() {
        return Err(anyhow!(
            "No compiler interface symbols found in dynamic library {}",
            path.display()
        ));
    }

    for symbol_name in super::external_interface::REQUIRED_EXTERNAL_SYMBOL_NAMES {
        if !discovered.contains_key(*symbol_name) {
            return Err(anyhow!(
                "Missing required symbol: {} in dynamic library {}",
                symbol_name,
                path.display()
            ));
        }
    }

    let mut symbol_map = HashMap::new();
    for (canonical_name, raw_name) in discovered {
        let symbol_ptr = unsafe { library.get(raw_name.as_str()) }
            .or_else(|_| unsafe { library.get(canonical_name.as_str()) })
            .with_context(|| {
                format!(
                    "Failed to resolve symbol '{}' from dynamic library {}",
                    canonical_name,
                    path.display()
                )
            })?;
        symbol_map.insert(canonical_name, symbol_ptr as usize);
    }

    Ok(symbol_map)
}

fn collect_file_symbols(
    file: &object::File<'_>,
    discovered: &mut BTreeMap<String, String>,
) -> Result<()> {
    for symbol in file.symbols() {
        collect_named_symbol(file, &symbol, discovered);
    }
    for symbol in file.dynamic_symbols() {
        collect_named_symbol(file, &symbol, discovered);
    }
    for export in file.exports()? {
        if let Ok(name) = std::str::from_utf8(export.name()) {
            record_symbol_name(name, discovered);
        }
    }
    Ok(())
}

fn collect_named_symbol<'data>(
    file: &object::File<'data>,
    symbol: &impl ObjectSymbol<'data>,
    discovered: &mut BTreeMap<String, String>,
) {
    if !symbol.is_definition() || symbol.kind() != SymbolKind::Text {
        return;
    }
    if let Some(section_index) = symbol.section_index() {
        let Ok(section) = file.section_by_index(section_index) else {
            return;
        };
        if !matches!(section.kind(), object::SectionKind::Text) {
            return;
        }
    }
    if let Ok(name) = symbol.name() {
        record_symbol_name(name, discovered);
    }
}

fn record_symbol_name(name: &str, discovered: &mut BTreeMap<String, String>) {
    if !is_external_symbol_name(name) {
        return;
    }

    let canonical_name = normalize_symbol_name(name).to_string();
    discovered
        .entry(canonical_name)
        .or_insert_with(|| name.to_string());
}
