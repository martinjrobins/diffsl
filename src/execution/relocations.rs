use std::{collections::HashMap, env::consts::ARCH};

use object::{
    elf::{
        R_386_16, R_386_8, R_386_PC16, R_386_PC8, R_X86_64_16, R_X86_64_32, R_X86_64_64,
        R_X86_64_8, R_X86_64_PC16, R_X86_64_PC32, R_X86_64_PC8, R_X86_64_PLT32,
    },
    macho::{ARM64_RELOC_PAGE21, ARM64_RELOC_PAGEOFF12},
    File, Object, ObjectSection, ObjectSymbol, Relocation, RelocationFlags, RelocationKind,
    RelocationTarget, Section,
};

use anyhow::{anyhow, Result};

use super::{compiler::MappedSection, functions::function_resolver};

/// https://blog.cloudflare.com/how-to-execute-an-object-file-part-3/
#[cfg(target_arch = "x86_64")]
#[repr(C)]
pub(crate) struct JumpTableEntry {
    addr: *const u8,
    instr: [u8; 6],
}

#[cfg(target_arch = "x86_64")]
impl JumpTableEntry {
    fn new(addr: *const u8) -> Self {
        let mut ret = Self {
            addr,
            // unconditional x64 JMP instruction
            // jmp    QWORD PTR [rip+0xfffffffffffffff2]
            // should always be {0xff, 0x25, 0xf2, 0xff, 0xff, 0xff}
            // so it would jump to an address stored at addr above
            instr: [0xFF, 0x25, 0xF2, 0xFF, 0xFF, 0xFF],
        };
        // just in case we'll calculate the offset and overwrite it
        let addr_ptr = &ret.addr as *const *const u8 as *const u8;
        let after_ptr = unsafe { (ret.instr.as_ptr()).add(ret.instr.len()) };
        let offset = (-((after_ptr as usize - addr_ptr as usize) as i32)).to_ne_bytes();
        ret.instr[2..6].copy_from_slice(&offset);
        ret
    }
    fn jump_ptr(&self) -> *const u8 {
        &self.instr[0] as *const u8
    }
    pub(crate) fn from_bytes(bytes: &mut [u8]) -> &mut [Self] {
        let size = std::mem::size_of::<Self>();
        let len = bytes.len() / size;
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut Self, len) }
    }
}

/// https://blog.cloudflare.com/how-to-execute-an-object-file-part-4/
#[cfg(target_arch = "aarch64")]
#[repr(C)]
pub(crate) struct JumpTableEntry {
    instr: [u32; 4],
}

#[cfg(target_arch = "aarch64")]
impl JumpTableEntry {
    fn new(addr: *const u8) -> Self {
        let addr = addr as u64;
        let mov = 0b11010010100000000000000000001001 | u32::try_from((addr << 48) >> 43).unwrap();
        let movk1 =
            0b11110010101000000000000000001001 | u32::try_from(((addr >> 16) << 48) >> 43).unwrap();
        let movk2 =
            0b11110010110000000000000000001001 | u32::try_from(((addr >> 32) << 48) >> 43).unwrap();

        // mov  x9, #0x0c14
        // movk x9, #0x5555, lsl #16
        // movk x9, #0x5555, lsl #3
        // br   x9
        Self {
            instr: [mov, movk1, movk2, 0xd61f0120],
        }
    }
    fn jump_ptr(&self) -> *const u8 {
        self.instr.as_ptr() as *const u8
    }
    pub(crate) fn from_bytes(bytes: &mut [u8]) -> &mut [Self] {
        let size = std::mem::size_of::<Self>();
        let len = bytes.len() / size;
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut Self, len) }
    }
}

/// returns the section of the relocation target symbol (if any)
pub(crate) fn relocation_target_section<'file, 'data>(
    file: &'file File<'data>,
    rela: &Relocation,
) -> Option<Section<'data, 'file>> {
    if let RelocationTarget::Symbol(symbol_index) = rela.target() {
        // Check if the symbol is defined in the current object file
        let symbol = file.symbol_by_index(symbol_index).unwrap();
        symbol
            .section_index()
            .map(|section_index| file.section_by_index(section_index).unwrap())
    } else {
        // The relocation target is not a symbol
        None
    }
}

pub(crate) fn is_jump_table_entry(file: &File<'_>, rela: &Relocation) -> bool {
    relocation_target_section(file, rela).is_none()
        && (rela.kind() == RelocationKind::PltRelative || rela.kind() == RelocationKind::Relative)
}

fn handle_relocation_elf_x86(
    s: *const u8,
    a: i64,
    p: *mut u8,
    r_type: u32,
    size: u8,
) -> Result<()> {
    let val = match r_type {
        // S + A
        R_386_16 | R_386_8 | R_X86_64_64 | R_X86_64_32 | R_X86_64_16 | R_X86_64_8 => {
            // also R_386_32
            i64::try_from(s as usize).unwrap() + a
        }
        // S + A - P
        R_386_PC16 | R_386_PC8 | R_X86_64_PC32 | R_X86_64_PC16 | R_X86_64_PC8 | R_X86_64_PLT32 => {
            // also R_386_PC32
            i64::try_from(s as usize).unwrap() + a - i64::try_from(p as usize).unwrap()
        }
        _ => {
            return Err(anyhow!(
                "Unsupported relocation type {:?} for elf x64",
                r_type
            ))
        }
    };
    match size {
        32 => unsafe { (p as *mut i32).write(i32::try_from(val).unwrap()) },
        64 => unsafe { (p as *mut i64).write(val) },
        _ => return Err(anyhow!("Unsupported relocation size {:?}", size)),
    }
    Ok(())
}

fn handle_relocation_macho_aarch64(
    s: *const u8,
    a: i64,
    p: *mut u8,
    r_type: u8,
    _r_pcrel: bool,
    _r_length: u8,
) -> Result<()> {
    match r_type {
        // offset within page, scaled by r_length
        ARM64_RELOC_PAGEOFF12 => {
            // https://blog.cloudflare.com/how-to-execute-an-object-file-part-4/§
            // The mask of `add` instruction to separate
            // opcode, registers and calculated value
            let mask_add: u32 = 0b11111111110000000000001111111111;
            // S + A
            let val = i64::try_from(s as usize).unwrap() + a;
            // shift left the calculated value by 10 bits and bitwise AND with the mask to get the lower 12 bits
            let val = ((val as u32) << 10) & !mask_add;

            let mut instr = unsafe { (p as *const u32).read() };
            // zero out the offset bits
            instr &= mask_add;
            // insert the calculated value
            instr |= val;
            // write the instruction back to the patch offset
            unsafe { (p as *mut u32).write(instr) };
        }
        // pc-rel distance to page of target
        ARM64_RELOC_PAGE21 => {
            // Page(S+A)-Page(P), Page(expr) is defined as (expr & ~0xFFF)
            let val = ((i64::try_from(s as usize).unwrap() + a) >> 12) - ((p as i64) >> 12);
            let val = u32::try_from(val).unwrap();
            // 2 low bits of immediate value are placed in the position 30:29 and the rest in the position 23:5.
            let immlo = (val & (0xf >> 2)) << 29;
            let immhi = (val & ((0xffffff >> 13) << 2)) << 22;
            unsafe {
                let instr = (p as *const u32).read();
                (p as *mut u32).write(instr | immlo | immhi);
            }
        }
        _ => {
            return Err(anyhow!(
                "Unsupported relocation type {:?} for macho aarch64",
                r_type
            ))
        }
    }
    Ok(())
}

fn relocation(rela: &Relocation, s: *const u8, p: *mut u8) -> Result<()> {
    let a = rela.addend();
    match rela.flags() {
        RelocationFlags::Elf { r_type } => match ARCH {
            "x86_64" => handle_relocation_elf_x86(s, a, p, r_type, rela.size()),
            "x86" => handle_relocation_elf_x86(s, a, p, r_type, rela.size()),
            _ => Err(anyhow!(
                "Unsupported architecture {} for ELF relocations",
                ARCH
            ))?,
        },
        RelocationFlags::MachO {
            r_type,
            r_pcrel,
            r_length,
        } => match ARCH {
            "aarch64" => handle_relocation_macho_aarch64(s, a, p, r_type, r_pcrel, r_length),
            _ => Err(anyhow!(
                "Unsupported architecture {} for MachO relocations",
                ARCH
            ))?,
        },
        _ => Err(anyhow!("Only ELF and MachO relocations are supported")),
    }
}

pub(crate) fn handle_relocation(
    file: &File<'_>,
    rela: &Relocation,
    p: *mut u8,
    mapped_sections: &HashMap<String, MappedSection>,
) -> Result<()> {
    let symbol_index = match rela.target() {
        RelocationTarget::Symbol(s) => s,
        _ => Err(anyhow!(
            "Only relocation targets that are symbols are supported"
        ))?,
    };
    let symbol = file.symbol_by_index(symbol_index).unwrap();
    let s = match symbol.section_index() {
        Some(section_index) => {
            let section = file.section_by_index(section_index).unwrap();
            let section_name = section.name().expect("Could not get section name");
            let section_ptr = mapped_sections[section_name].as_ptr();
            unsafe { section_ptr.offset(symbol.address() as isize) }
        }
        None => {
            // must be an external function call generate the jump table entry
            function_resolver(symbol.name().unwrap())
                .unwrap_or_else(|| panic!("Could not resolve function {}", symbol.name().unwrap()))
        }
    };
    relocation(rela, s, p)
}

pub(crate) fn handle_jump_entry(
    file: &File<'_>,
    rela: &Relocation,
    p: *mut u8,
    jumptable_entry: &mut JumpTableEntry,
) -> Result<()> {
    let symbol_index = match rela.target() {
        RelocationTarget::Symbol(s) => s,
        _ => Err(anyhow!(
            "Only relocation targets that are symbols are supported"
        ))?,
    };
    let symbol = file.symbol_by_index(symbol_index).unwrap();
    let symbol_name = symbol.name().unwrap();
    // must be an external function call generate the jump table entry
    let addr = function_resolver(symbol_name)
        .unwrap_or_else(|| panic!("Could not resolve function {}", symbol_name));
    *jumptable_entry = JumpTableEntry::new(addr);
    let s = jumptable_entry.jump_ptr();
    relocation(rela, s, p)
}
