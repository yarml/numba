use std::{
  marker::PhantomData,
  mem::{self, MaybeUninit},
};

extern "C" {
  fn numba_Allocate(s: usize) -> *const ();
  fn numba_Free(ptr: *const ());
  fn numba_CopyToDevice(dev_dst: *const (), host_src: *const (), s: usize);
  fn numba_CopyToHost(host_dst: *mut (), dev_src: *const (), s: usize);
  fn numba_CopyDeviceToDevice(dev_dst: *const (), dev_src: *const (), s: usize);
  fn numba_Memset(dev_dst: *const (), value: i32, s: usize);
}

#[derive(Debug)]
pub struct CudaBox<T: ?Sized> {
  ptr: *const (),
  size: usize,
  phantom: PhantomData<T>,
}

impl<T> CudaBox<T> {
  pub fn new(x: T) -> Self {
    let size = mem::size_of::<T>();
    let ptr = unsafe { numba_Allocate(size) };
    unsafe { numba_CopyToDevice(ptr, &x as *const T as *const (), size) };
    CudaBox {
      ptr,
      size,
      phantom: PhantomData,
    }
  }
  pub fn to_host(&self) -> T {
    let mut x = MaybeUninit::uninit();
    unsafe { numba_CopyToHost(x.as_mut_ptr() as *mut (), self.ptr, self.size) };
    unsafe { x.assume_init() }
  }
}

impl<T: ?Sized> CudaBox<T> {
  pub fn to_device(&self, x: Box<T>) {
    unsafe {
      numba_CopyToDevice(
        self.ptr,
        x.as_ref() as *const T as *const (),
        self.size,
      )
    };
  }

  pub fn as_ptr(&self) -> *const () {
    self.ptr
  }
  pub fn size(&self) -> usize {
    self.size
  }
}

impl<T> CudaBox<[T]> {
  pub fn new_zeroed_slice(len: usize) -> Self {
    let size = mem::size_of::<T>() * len;
    let ptr = unsafe { numba_Allocate(size) };
    unsafe { numba_Memset(ptr, 0, size) };
    CudaBox {
      ptr,
      size,
      phantom: PhantomData,
    }
  }
}

impl<T: Copy> CudaBox<[T]> {
  pub fn to_host_slice(&self) -> Box<[T]> {
    let mut x = vec![MaybeUninit::uninit(); self.size / mem::size_of::<T>()];
    unsafe {
      numba_CopyToHost(x.as_mut_ptr() as *mut (), self.ptr, self.size);
    }
    x.into_iter().map(|x| unsafe { x.assume_init() }).collect()
  }
}

impl<T: ?Sized> Drop for CudaBox<T> {
  fn drop(&mut self) {
    unsafe { numba_Free(self.ptr) };
  }
}

impl<T: Default> Default for CudaBox<T> {
  fn default() -> Self {
    CudaBox::new(T::default())
  }
}

impl<T: ?Sized> Clone for CudaBox<T> {
  fn clone(&self) -> Self {
    let ptr = unsafe { numba_Allocate(self.size) };
    unsafe { numba_CopyDeviceToDevice(ptr, self.ptr, self.size) };
    CudaBox {
      ptr,
      size: self.size,
      phantom: PhantomData,
    }
  }
  fn clone_from(&mut self, source: &Self) {
    unsafe { numba_CopyDeviceToDevice(self.ptr, source.ptr, self.size) };
  }
}

impl<I> FromIterator<I> for CudaBox<[I]> {
  fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
    let mut collection: Vec<_> = iter.into_iter().collect();
    collection.shrink_to_fit();
    let elsize = mem::size_of::<I>();
    let size = collection.len() * elsize;
    let ptr = unsafe { numba_Allocate(size) };
    unsafe { numba_CopyToDevice(ptr, collection.as_ptr() as *const (), size) };
    CudaBox {
      ptr,
      size,
      phantom: PhantomData,
    }
  }
}
