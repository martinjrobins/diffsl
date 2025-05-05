use anyhow::{anyhow, Result};
use mmap_rs::MmapOptions;
use std::collections::HashMap;

use super::{
    mmap::MappedSection,
    module::{CodegenModule, CodegenModuleJit, CodegenModuleLink},
    relocations::{
        handle_jump_entry, handle_relocation, is_jump_table_entry, relocation_target_section,
        symbol_offset, JumpTableEntry,
    },
};
use object::{Object, ObjectSection, ObjectSymbol, SectionKind};

pub struct ObjectModule {
    sections: HashMap<String, MappedSection>,
    symbols: HashMap<String, isize>,
    code_section_name: String,
}

impl CodegenModule for ObjectModule {}

impl CodegenModuleLink for ObjectModule {
    fn from_object(buffer: &[u8]) -> Result<Self> {
        let mut mapped_sections = HashMap::new();

        let file = object::File::parse(buffer)?;
        let mut text_sec = None;
        for section in file.sections() {
            if let SectionKind::Text = section.kind() {
                if text_sec.is_some() {
                    return Err(anyhow!("Multiple .text sections found"));
                }
                text_sec = Some(section);
            }
        }

        let text_sec = text_sec.ok_or_else(|| anyhow!("Could not find .text section"))?;
        let min_page_size = MmapOptions::page_size();
        let mut sections_to_map = HashMap::new();
        sections_to_map.insert(
            text_sec.name().expect("Could not get section name"),
            text_sec.index(),
        );

        let jumptable_entry_size = std::mem::size_of::<JumpTableEntry>();

        let mut jumptable_num_entries: usize = 0;
        let sections_to_map = text_sec
            .relocations()
            .fold(sections_to_map, |mut acc, (_, rela)| {
                if let Some(section) = relocation_target_section(&file, &rela) {
                    if section.index() == text_sec.index() {
                        // skip the text section
                        return acc;
                    }
                    let section_name = section.name().expect("Could not get section name");
                    acc.insert(section_name, section.index());
                }
                if is_jump_table_entry(&file, &rela) {
                    // this is a jump table entry
                    jumptable_num_entries += 1;
                }
                acc
            });
        let jumptable_size =
            (jumptable_num_entries * jumptable_entry_size).div_ceil(min_page_size) * min_page_size;
        let mapped_sizes = sections_to_map
            .values()
            .map(|section| {
                let section = file
                    .section_by_index(*section)
                    .expect("Could not find section");
                let section_size = section.size() as usize;
                // round up to minimum page size
                section_size.div_ceil(min_page_size) * min_page_size
            })
            .collect::<Vec<_>>();
        let mut map =
            MmapOptions::new(mapped_sizes.iter().sum::<usize>() + jumptable_size + min_page_size)
                .unwrap()
                .map()
                .unwrap();
        for ((name, section_id), size) in sections_to_map.iter().zip(mapped_sizes.iter()) {
            let section = file
                .section_by_index(*section_id)
                .expect("Could not find section");
            let mut section_map = map.split_to(*size).unwrap().make_mut().unwrap();
            let section_data = section.data().expect("Could not get section data");
            section_map.as_mut_slice()[..section_data.len()].copy_from_slice(section_data);
            match section.kind() {
                SectionKind::Data
                | SectionKind::Text
                | SectionKind::Common
                | SectionKind::UninitializedData
                | SectionKind::UninitializedTls => {
                    // writable data needs to be writable
                    mapped_sections.insert(name.to_string(), MappedSection::Mutable(section_map));
                }
                _ => {
                    // everything else is immutable
                    mapped_sections.insert(
                        name.to_string(),
                        MappedSection::Immutable(section_map.make_read_only().unwrap()),
                    );
                }
            }
        }
        let mut jumptable_map = if jumptable_size > 0 {
            Some(map.split_to(jumptable_size).unwrap().make_mut().unwrap())
        } else {
            None
        };
        let mut jumptable_idx = 0;
        let mut jumptable = jumptable_map
            .as_mut()
            .map(|jumptable_map| JumpTableEntry::from_bytes(jumptable_map.as_mut_slice()));
        for (offset, rela) in text_sec.relocations() {
            let text_ptr = mapped_sections
                .get_mut(text_sec.name().unwrap())
                .unwrap()
                .as_mut_ptr()
                .unwrap();
            let patch_ptr = unsafe { text_ptr.offset(offset as isize) };
            if is_jump_table_entry(&file, &rela) {
                let jumptable_entry = &mut jumptable.as_mut().unwrap()[jumptable_idx];
                handle_jump_entry(&file, &rela, patch_ptr, jumptable_entry)?;
                jumptable_idx += 1;
            } else {
                handle_relocation(&file, &rela, patch_ptr, &mapped_sections)?;
            }
        }
        // make jumptable read only and executable and insert it into the mapped sections
        if let Some(jumptable_map) = jumptable_map {
            mapped_sections.insert(
                "jumptable".to_string(),
                MappedSection::Immutable(
                    jumptable_map.make_read_only().unwrap().make_exec().unwrap(),
                ),
            );
        }
        // make text section immutable and executable
        let mut text_map = mapped_sections.remove(text_sec.name().unwrap()).unwrap();
        text_map = text_map.make_read_only().unwrap().make_exec().unwrap();
        mapped_sections.insert(text_sec.name().unwrap().to_string(), text_map);

        let text_sec_name = text_sec.name().unwrap();
        let mut symbol_map = HashMap::new();
        for symbol in file.symbols() {
            if let Ok(name) = symbol.name() {
                if let Some(section_index) = symbol.section_index() {
                    let section = file
                        .section_by_index(section_index)
                        .expect("Could not find section");
                    if let SectionKind::Text = section.kind() {
                        let offset = symbol_offset(&file, &symbol, &section)?;
                        // for some reason on macOS the symbol name is prefixed with an underscore, remove it
                        let name = name.strip_prefix("_").unwrap_or(name);
                        symbol_map.insert(name.to_string(), offset);
                        // skip text sections, they are already mapped
                    }
                }
            }
        }

        Ok(Self {
            sections: mapped_sections,
            symbols: symbol_map,
            code_section_name: text_sec_name.to_string(),
        })
    }
}

impl CodegenModuleJit for ObjectModule {
    fn jit(&mut self) -> Result<HashMap<String, *const u8>> {
        // map symbols to addresses
        let mut symbol_map = HashMap::new();
        for (name, offset) in &self.symbols {
            let section = self.sections.get(self.code_section_name.as_str()).unwrap();
            let func_ptr = unsafe { section.as_ptr().offset(*offset) };
            symbol_map.insert(name.clone(), func_ptr);
        }
        Ok(symbol_map)
    }
}
