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

static __global__ void kernel_linearSigmoid(float* a, float* b, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    b[index] = 1 / (1 + exp(-a[index]));
  }
}

extern "C" {
void numba_VecAdd(float* devA, float* devB, float* devC, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_linearAdd<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, N);
  cudaDeviceSynchronize();
}

void numba_MatrixMul(
  float* devA,
  float* devB,
  float* devC,
  size_t N,
  size_t M,
  size_t R
) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(
    (R + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y
  );
  kernel_matrixMul<<<blocksPerGrid, threadsPerBlock>>>(
    devA,
    devB,
    devC,
    N,
    M,
    R
  );
  cudaDeviceSynchronize();
}

void numba_VecSigmoid(float* devA, float* devB, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_linearSigmoid<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, N);
  cudaDeviceSynchronize();
}
}
