use std::{collections::HashMap, fs, path::PathBuf, time::Instant};

extern "C" {
  fn numba_BenchmarkAllocate(ptrs: *mut *const (), s: usize, n: usize);
  fn numba_BenchmarkFree(ptrs: *const *const (), n: usize);
}

fn load_allocations(path: PathBuf) -> HashMap<usize, usize> {
  let mut allocations = HashMap::new();
  fs::read_to_string(path)
    .expect("Failed to read allocations file")
    .lines()
    .for_each(|s| {
      let alloc_size = s.parse().expect("Failed to parse allocation size");
      if let Some(num_alloc) = allocations.get_mut(&alloc_size) {
        *num_alloc += 1;
        return;
      } else {
        allocations.insert(alloc_size, 0);
      }
    });
  allocations
}

fn main() {
  let alloc_per_size = 100000; 
  let alloc_sizes = load_allocations("test/test.train".into());
  let mut ptrs = vec![0usize; alloc_per_size];

  let mut total_saved_time = 0.;

  for (alloc_size, num_alloc) in alloc_sizes.iter() {
    let alloc_start = Instant::now();
    unsafe {
      numba_BenchmarkAllocate(
        ptrs.as_mut_ptr() as *mut *const (),
        *alloc_size,
        alloc_per_size,
      );
    }
    let alloc_end = Instant::now();

    let time_per_buffer = alloc_end.duration_since(alloc_start).as_micros()
      as f32
      / alloc_per_size as f32;
    let saved_alloc = time_per_buffer * *num_alloc as f32;

    println!(
      "Allocated {} buffer of size {} in {}ms (Averaging {}us per buffer), there are {} of them, taking out these allocations may save {}us",
      alloc_per_size,
      alloc_size,
      alloc_end.duration_since(alloc_start).as_millis(),
      time_per_buffer,
      num_alloc,
      saved_alloc,
    );
    let free_start = Instant::now();
    unsafe {
      numba_BenchmarkFree(ptrs.as_ptr() as *const *const (), alloc_per_size);
    }
    let free_end = Instant::now();
    let timer_per_buffer = free_end.duration_since(free_start).as_micros()
      as f32
      / alloc_per_size as f32;
    let saved_free = timer_per_buffer * *num_alloc as f32;
    println!(
      "Freed {} buffer of size {} in {}ms (Averaging {}us per buffer), taking out these allocations may save {}us",
      alloc_per_size,
      alloc_size,
      free_end.duration_since(free_start).as_millis(),
      timer_per_buffer,
      saved_free
    );
    total_saved_time += saved_alloc + saved_free;
  }

  println!("Total time that could be saved if all allocations are removed: {}us ({}s)", total_saved_time, total_saved_time / 1_000_000.0);
}
