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

NumbaMatrix &NumbaMatrix::operator=(NumbaMatrix &&other) {
  if (this->elements) {
    delete[] this->elements;
  }
  nRows = other.nRows;
  nCols = other.nCols;
  elements = other.elements;
  other.elements = nullptr;
  return *this;
}

pair<int, int> NumbaMatrix::shape() const { return make_pair(nRows, nCols); }
float &NumbaMatrix::at(int i, int j) const { return elements[i * nCols + j]; }

NumbaMatrix NumbaMatrix::add(NumbaMatrix const &bMat) const {
  assert(shape() == bMat.shape());

  float *a, *b, *c;
  size_t size = nRows * nCols * sizeof(float);

  cout << "bye" << endl;
  exit(0);
  gpuErrchk(cudaMalloc(&a, size));
  gpuErrchk(cudaMalloc(&b, size));
  gpuErrchk(cudaMalloc(&c, size));

  gpuErrchk(cudaMemcpy(a, elements, size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(b, bMat.elements, size, cudaMemcpyHostToDevice));

  addVec<<<(nRows * nCols + 255) / 256, 256>>>(a, b, c, nRows * nCols);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  NumbaMatrix c_mat(nRows, nCols);
  gpuErrchk(cudaMemcpy(c_mat.elements, c, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(a));
  gpuErrchk(cudaFree(b));
  gpuErrchk(cudaFree(c));

  return move(c_mat);
}
NumbaMatrix NumbaMatrix::mul(NumbaMatrix const &bMat) const {
  assert(nCols == bMat.nRows);

  float *a, *b, *c;
  size_t size = nRows * nCols * sizeof(float);

  gpuErrchk(cudaMalloc(&a, size));
  gpuErrchk(cudaMalloc(&b, size));
  gpuErrchk(cudaMalloc(&c, size));

  gpuErrchk(cudaMemcpy(a, elements, size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(b, bMat.elements, size, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((nRows + 15) / 16, (bMat.nCols + 15) / 16);
  mulMat<<<numBlocks, threadsPerBlock>>>(a, b, c, nRows, bMat.nCols);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  NumbaMatrix c_mat(nRows, bMat.nCols);
  gpuErrchk(cudaMemcpy(c_mat.elements, c, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(a));
  gpuErrchk(cudaFree(b));
  gpuErrchk(cudaFree(c));

  return move(c_mat);
}
NumbaMatrix NumbaMatrix::scale(float scale) const {
  float *a, *b;
  size_t size = nRows * nCols * sizeof(float);

  gpuErrchk(cudaMalloc(&a, size));
  gpuErrchk(cudaMalloc(&b, size));

  cudaMemcpy(a, elements, size, cudaMemcpyHostToDevice);

  scaleMat<<<(nRows * nCols + 255) / 256, 256>>>(a, b, nRows * nCols, scale);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  NumbaMatrix bMat(nRows, nCols);
  cudaMemcpy(bMat.elements, b, size, cudaMemcpyDeviceToHost);

  gpuErrchk(cudaFree(a));
  gpuErrchk(cudaFree(b));

  return move(bMat);
}
NumbaMatrix NumbaMatrix::linearize() const {
  NumbaMatrix result(nRows * nCols, 1);
  memcpy(result.elements, elements, nRows * nCols * sizeof(float));
  return move(result);
}

float &NumbaMatrix::operator()(int i, int j) const { return at(i, j); }
NumbaMatrix NumbaMatrix::operator+(NumbaMatrix const &bMat) const {
  return move(add(bMat));
}
NumbaMatrix NumbaMatrix::operator*(NumbaMatrix const &bMat) const {
  return move(mul(bMat));
}
NumbaMatrix NumbaMatrix::operator*(float s) const { return move(scale(s)); }
