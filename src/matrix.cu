static __global__ void
kernel_linearAdd(float* a, float* b, float* c, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    c[index] = a[index] + b[index];
  }
}

// Matrix multiplication kernel:
// Path: src/matrix.cu
static __global__ void
kernel_matrixMul(float* a, float* b, float* c, size_t N, size_t M, size_t R) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < R) {
    float sum = 0;
    for (int i = 0; i < M; i++) {
      sum += a[row * M + i] * b[i * R + col];
    }
    c[row * R + col] = sum;
  }
}

extern "C" {
void linear_add(float* a, float* b, float* c, size_t N) {
  size_t size = N * sizeof(float);
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_linearAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void matrix_mul(float* a, float* b, float* c, size_t N, size_t M, size_t R) {
  size_t sizeA = N * M * sizeof(float);
  size_t sizeB = M * R * sizeof(float);
  size_t sizeC = N * R * sizeof(float);
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sizeA);
  cudaMalloc(&d_b, sizeB);
  cudaMalloc(&d_c, sizeC);
  cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(
    (R + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y
  );
  kernel_matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, M, R);
  cudaMemcpy(c, d_c, sizeC, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
}
