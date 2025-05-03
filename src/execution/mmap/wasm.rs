struct Mmap(&'static [u8]);
struct MmapMut(&'static mut [u8]);
struct MmapOptions {
    size: usize,
}

impl MmapOptions {
    /// for wasm32 the page size is fixed at 64KiB.
    pub fn page_size() -> usize {
        64 * 1024
    }
    
    pub fn new(size: usize) -> Self {
        // panic if size is not a multiple of the page size
        let page_size = Self::page_size();
        if size % page_size != 0 {
            panic!("Size must be a multiple of the page size");
        }
        Self { size }
    }
    
    pub fn map(&self) -> Result<Mmap> {
        Mmap::new(self.size);
    }
}

impl Mmap {
    pub unsafe fn from_raw_parts(ptr: *const u8, size: usize) -> Result<Mmap> {
        if ptr.is_null() {
            return Err(anyhow!("Pointer is null"));
        }
        Ok(Mmap(std::slice::from_raw_parts(ptr, size)))
    }
    pub fn new(size: usize) -> Result<Mmap> {
        let ptr = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(size, page_size).unwrap()) };
        if ptr.is_null() {
            return Err(anyhow!("Failed to allocate memory"));
        }
        Ok(Mmap(unsafe { std::slice::from_raw_parts(ptr, size) }))
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
    pub fn make_mut(self) -> Result<MmapMut> {
        unsafe { MmapMut::from_raw_parts(self.0.as_ptr() as *mut u8, self.0.len()) }
    }
    pub fn split_to(&mut self, offset: usize) -> Result<Mmap> {
        if offset > self.0.len() {
            return Err(anyhow!("Offset is out of bounds"));
        }
        let (left, right) = self.0.split_at(offset);
        self.0 = left;
        Ok(Mmap(right))
    }
}

impl MmapMut {
    pub unsafe fn from_raw_parts(ptr: *mut u8, size: usize) -> Result<MmapMut> {
        if ptr.is_null() {
            return Err(anyhow!("Pointer is null"));
        }
        Ok(MmapMut(std::slice::from_raw_parts_mut(ptr, size)))
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }
    
    pub fn make_read_only(self) -> Result<Mmap> {
        unsafe { Mmap::from_raw_parts(self.0.as_ptr() as *const u8, self.0.len()) }
    }
    
    pub fn make_exec(self) -> Result<Mmap> {
        Ok(self)
    }
}