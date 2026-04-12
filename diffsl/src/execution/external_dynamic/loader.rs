use std::{ffi::CString, path::Path};

use anyhow::{anyhow, Context, Result};

#[cfg(all(unix, not(target_arch = "wasm32")))]
pub(super) struct DynamicLibrary {
    handle: *mut libc::c_void,
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
impl DynamicLibrary {
    pub(super) fn open(path: &Path) -> Result<Self> {
        use std::os::unix::ffi::OsStrExt;

        let path_cstr = CString::new(path.as_os_str().as_bytes()).with_context(|| {
            format!(
                "Dynamic library path contains an interior NUL byte: {}",
                path.display()
            )
        })?;

        let handle = unsafe {
            clear_dlerror();
            libc::dlopen(path_cstr.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL)
        };
        if handle.is_null() {
            return Err(anyhow!(
                "Failed to load dynamic library {}: {}",
                path.display(),
                dlerror_message().unwrap_or_else(|| "unknown loader error".to_string())
            ));
        }
        Ok(Self { handle })
    }

    pub(super) unsafe fn get(&self, name: &str) -> Result<*const u8> {
        let symbol_cstr = CString::new(name)
            .with_context(|| format!("Symbol name contains an interior NUL byte: {}", name))?;
        clear_dlerror();
        let symbol = libc::dlsym(self.handle, symbol_cstr.as_ptr());
        if symbol.is_null() {
            return Err(anyhow!(
                "{}",
                dlerror_message().unwrap_or_else(|| format!("symbol '{}' not found", name))
            ));
        }
        Ok(symbol.cast())
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                libc::dlclose(self.handle);
            }
        }
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
unsafe impl Send for DynamicLibrary {}

#[cfg(all(unix, not(target_arch = "wasm32")))]
unsafe impl Sync for DynamicLibrary {}

#[cfg(all(unix, not(target_arch = "wasm32")))]
unsafe fn clear_dlerror() {
    let _ = libc::dlerror();
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn dlerror_message() -> Option<String> {
    let err = unsafe { libc::dlerror() };
    if err.is_null() {
        return None;
    }
    Some(
        unsafe { std::ffi::CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(all(windows, not(target_arch = "wasm32")))]
pub(super) struct DynamicLibrary {
    handle: windows_sys::Win32::Foundation::HMODULE,
}

#[cfg(all(windows, not(target_arch = "wasm32")))]
impl DynamicLibrary {
    pub(super) fn open(path: &Path) -> Result<Self> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;

        let wide_path = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let handle = unsafe { LoadLibraryW(wide_path.as_ptr()) };
        if handle.is_null() {
            return Err(anyhow!(
                "Failed to load dynamic library {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { handle })
    }

    pub(super) unsafe fn get(&self, name: &str) -> Result<*const u8> {
        use windows_sys::Win32::System::LibraryLoader::GetProcAddress;

        let symbol_cstr = CString::new(name)
            .with_context(|| format!("Symbol name contains an interior NUL byte: {}", name))?;
        match GetProcAddress(self.handle, symbol_cstr.as_ptr().cast()) {
            Some(symbol) => Ok(symbol as *const u8),
            None => Err(anyhow!(
                "symbol '{}' not found: {}",
                name,
                std::io::Error::last_os_error()
            )),
        }
    }
}

#[cfg(all(windows, not(target_arch = "wasm32")))]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::FreeLibrary;

        unsafe {
            FreeLibrary(self.handle);
        }
    }
}

#[cfg(all(windows, not(target_arch = "wasm32")))]
unsafe impl Send for DynamicLibrary {}

#[cfg(all(windows, not(target_arch = "wasm32")))]
unsafe impl Sync for DynamicLibrary {}

#[cfg(any(target_arch = "wasm32", not(any(unix, windows))))]
pub(super) struct DynamicLibrary;

#[cfg(any(target_arch = "wasm32", not(any(unix, windows))))]
impl DynamicLibrary {
    pub(super) fn open(path: &Path) -> Result<Self> {
        let _ = path;
        Err(anyhow!(
            "Runtime dynamic library loading is not supported on this target"
        ))
    }

    pub(super) unsafe fn get(&self, name: &str) -> Result<*const u8> {
        let _ = name;
        Err(anyhow!(
            "Runtime dynamic library loading is not supported on this target"
        ))
    }
}
