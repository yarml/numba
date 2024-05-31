fn main() {
  cc::Build::new()
    .cuda(true)
    .flag("-gencode")
    .flag("arch=compute_80,code=sm_80")
    .flag("-t0")
    .include("inc/")
    .file("src/matrix.cu")
    .compile("test");
  println!("cargo:rerun-if-changed=src/matrix.cu");
}

