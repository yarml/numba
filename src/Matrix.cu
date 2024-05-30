#include <Matrix.hh>
#include <cassert>
#include <err.cuh>

using namespace std;

static __global__ void addVec(float *a, float *b, float *c, size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    c[index] = a[index] + b[index];
  }
}
static __global__ void mulMat(float *a, float *b, float *c, size_t N,
                              size_t M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < M) {
    float sum = 0;
    for (int k = 0; k < N; k++) {
      sum += a[i * N + k] * b[k * M + j];
    }
    c[i * M + j] = sum;
  }
}

__global__ void scaleMat(float *a, float *b, size_t N, float scale) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    b[index] = a[index] * scale;
  }
}
NumbaMatrix::NumbaMatrix(size_t rows, size_t cols) {
  this->nRows = rows;
  this->nCols = cols;
  this->elements = new float[rows * cols];
}
NumbaMatrix::NumbaMatrix(NumbaMatrix &&other) {
  nRows = other.nRows;
  nCols = other.nCols;
  elements = other.elements;
  other.elements = nullptr;
}
NumbaMatrix::~NumbaMatrix() {
  if (elements) {
    delete[] elements;
  }
}

pair<int, int> NumbaMatrix::shape() const { return make_pair(nRows, nCols); }
float &NumbaMatrix::at(int i, int j) const { return elements[i * nCols + j]; }

NumbaMatrix NumbaMatrix::add(NumbaMatrix const &b_mat) const {
  assert(shape() == b_mat.shape());

  float *a, *b, *c;
  size_t size = nRows * nCols * sizeof(float);

  cudaMalloc(&a, size);
  cudaMalloc(&b, size);
  cudaMalloc(&c, size);

  cudaMemcpy(a, elements, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b, b_mat.elements, size, cudaMemcpyHostToDevice);

  addVec<<<(nRows * nCols + 255) / 256, 256>>>(a, b, c, nRows * nCols);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  NumbaMatrix c_mat(nRows, nCols);
  cudaMemcpy(c_mat.elements, c, size, cudaMemcpyDeviceToHost);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return c_mat;
}
NumbaMatrix NumbaMatrix::mul(NumbaMatrix const &b_mat) const {
  assert(nCols == b_mat.nRows);

  float *a, *b, *c;
  size_t size = nRows * nCols * sizeof(float);

  cudaMalloc(&a, size);
  cudaMalloc(&b, size);
  cudaMalloc(&c, size);

  cudaMemcpy(a, elements, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b, b_mat.elements, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((nRows + 15) / 16, (b_mat.nCols + 15) / 16);
  mulMat<<<numBlocks, threadsPerBlock>>>(a, b, c, nRows, b_mat.nCols);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  NumbaMatrix c_mat(nRows, b_mat.nCols);
  cudaMemcpy(c_mat.elements, c, size, cudaMemcpyDeviceToHost);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return c_mat;
}
NumbaMatrix NumbaMatrix::scale(float scale) const {
  float *a, *b;
  size_t size = nRows * nCols * sizeof(float);

  cudaMalloc(&a, size);
  cudaMalloc(&b, size);

  cudaMemcpy(a, elements, size, cudaMemcpyHostToDevice);

  scaleMat<<<(nRows * nCols + 255) / 256, 256>>>(a, b, nRows * nCols, scale);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  NumbaMatrix b_mat(nRows, nCols);
  cudaMemcpy(b_mat.elements, b, size, cudaMemcpyDeviceToHost);

  cudaFree(a);
  cudaFree(b);

  return b_mat;
}

float &NumbaMatrix::operator()(int i, int j) const { return at(i, j); }
NumbaMatrix NumbaMatrix::operator+(NumbaMatrix const &b_mat) const {
  return add(b_mat);
}
NumbaMatrix NumbaMatrix::operator*(NumbaMatrix const &b_mat) const {
  return mul(b_mat);
}
NumbaMatrix NumbaMatrix::operator*(float s) const { return scale(s); }
