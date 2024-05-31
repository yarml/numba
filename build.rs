fn main() {
  let cuda_src = ["src/matrix.cu", "src/cuda.cu"];
  let mut cfg = cc::Build::new();
  cfg.cuda(true);
  cfg.flag("-gencode").flag("arch=compute_80,code=sm_80");
  cfg.flag("-t0");
  cfg.include("inc/");

  for src in cuda_src.iter() {
    cfg.file(src);
  }

  cfg.compile("numba");

  for src in cuda_src.iter() {
    println!("cargo:rerun-if-changed={}", src);
  }
}
