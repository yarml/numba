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

public:
  NumbaMatrix &operator=(NumbaMatrix const &) = delete;
  NumbaMatrix(NumbaMatrix const &) = delete;
  NumbaMatrix() = default;

public:
  std::pair<int, int> shape() const;
  float &at(int i, int j) const;

public:
  NumbaMatrix add(NumbaMatrix const &b_mat) const;
  NumbaMatrix mul(NumbaMatrix const &b_mat) const;
  NumbaMatrix scale(float scale) const;

public:
  float &operator()(int i, int j) const;
  NumbaMatrix operator+(NumbaMatrix const &b_mat) const;
  NumbaMatrix operator*(NumbaMatrix const &b_mat) const;
  NumbaMatrix operator*(float s) const;
};
