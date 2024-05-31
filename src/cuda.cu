/// CUDA API functions with a nice interface for Rust

extern "C" {
void* numba_Allocate(size_t s) {
  void* ptr;
  cudaMalloc(&ptr, s);
  return ptr;
}

void numba_Free(void* ptr) {
  cudaFree(ptr);
}

void numba_CopyToDevice(void* devDst, void* hostSrc, size_t s) {
  cudaMemcpy(devDst, hostSrc, s, cudaMemcpyHostToDevice);
}

void numba_CopyToHost(void* hostDst, void* devSrc, size_t s) {
  cudaMemcpy(hostDst, devSrc, s, cudaMemcpyDeviceToHost);
}

void numba_CopyDeviceToDevice(void* devDst, void* devSrc, size_t s) {
  cudaMemcpy(devDst, devSrc, s, cudaMemcpyDeviceToDevice);
}

void numba_Memset(void* devPtr, int value, size_t s) {
  cudaMemset(devPtr, value, s);
}
}