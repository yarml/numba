#pragma once

#include <cstddef>
#include <utility>

class NumbaMatrix {
private:
  size_t nRows;
  size_t nCols;
  float *elements;

public:
  NumbaMatrix(size_t rows, size_t cols);
  NumbaMatrix(NumbaMatrix &&other);
  ~NumbaMatrix();
  NumbaMatrix &operator=(NumbaMatrix &&other);

public:
  NumbaMatrix(NumbaMatrix const &) = delete;
  NumbaMatrix &operator=(NumbaMatrix const &) = delete;

public:
  std::pair<int, int> shape() const;
  float &at(int i, int j) const;

public:
  NumbaMatrix add(NumbaMatrix const &bMat) const;
  NumbaMatrix mul(NumbaMatrix const &bMat) const;
  NumbaMatrix scale(float scale) const;
  NumbaMatrix linearize() const;

public:
  float &operator()(int i, int j) const;
  NumbaMatrix operator+(NumbaMatrix const &bMat) const;
  NumbaMatrix operator*(NumbaMatrix const &bMat) const;
  NumbaMatrix operator*(float s) const;
};
