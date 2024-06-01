static __global__ void kernel_VecAdd(float* a, float* b, float* c, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    c[index] = a[index] + b[index];
  }
}

static __global__ void kernel_VecSub(float* a, float* b, float* c, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    c[index] = a[index] - b[index];
  }
}

static __global__ void kernel_VecMul(float* a, float* b, float* c, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    c[index] = a[index] * b[index];
  }
}

static __global__ void
kernel_MatrixMul(float* a, float* b, float* c, size_t N, size_t M, size_t R) {
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

static __global__ void kernel_VecSigmoid(float* a, float* b, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    b[index] = 1 / (1 + exp(-a[index]));
  }
}

static __global__ void kernel_VecSigmoidPrime(float* a, float* b, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    float sigmoid = 1 / (1 + exp(-a[index]));
    b[index] = sigmoid * (1 - sigmoid);
  }
}

static __global__ void kernel_VecDot(float* a, float* b, float* c, size_t N) {
  __shared__ float cache[256];
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  float sum = 0;
  while (index < N) {
    sum += a[index] * b[index];
    index += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = sum;
  __syncthreads();
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0)
    c[blockIdx.x] = cache[0];
}

static __global__ void
kernel_MatrixTranspose(float* a, float* b, size_t N, size_t M) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < M) {
    b[col * N + row] = a[row * M + col];
  }
}

static __global__ void kernel_VecScale(float* a, float* b, float s, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    b[index] = a[index] * s;
  }
}

extern "C" {
void numba_VecAdd(float* devA, float* devB, float* devC, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_VecAdd<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, N);
  cudaDeviceSynchronize();
}

void numba_VecSub(float* devA, float* devB, float* devC, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_VecSub<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, N);
  cudaDeviceSynchronize();
}

void numba_VecMul(float* devA, float* devB, float* devC, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_VecMul<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, N);
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
  kernel_MatrixMul<<<blocksPerGrid, threadsPerBlock>>>(
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
  kernel_VecSigmoid<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, N);
  cudaDeviceSynchronize();
}

void numba_VecSigmoidPrime(float* devA, float* devB, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_VecSigmoidPrime<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, N);
  cudaDeviceSynchronize();
}

float numba_VecDot(float* devA, float* devB, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  float* devCFragments;
  cudaMalloc(&devCFragments, sizeof(float) * blocksPerGrid);
  kernel_VecDot<<<blocksPerGrid, threadsPerBlock>>>(
    devA,
    devB,
    devCFragments,
    N
  );
  cudaDeviceSynchronize();
  float hostCFragments[blocksPerGrid];
  cudaMemcpy(
    hostCFragments,
    devCFragments,
    sizeof(float) * blocksPerGrid,
    cudaMemcpyDeviceToHost
  );
  float result = 0;
  for (int i = 0; i < blocksPerGrid; i++) {
    result += hostCFragments[i];
  }
  cudaFree(devCFragments);
  return result;
}

void numba_MatrixTranspose(float* devA, float* devB, size_t N, size_t M) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(
    (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y
  );
  kernel_MatrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, N, M);
  cudaDeviceSynchronize();
}

void numba_VecScale(float* devA, float* devB, float s, size_t N) {
  size_t threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_VecScale<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, s, N);
  cudaDeviceSynchronize();
}
}