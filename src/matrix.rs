use std::ops::{Add, Index, IndexMut, Mul};

extern "C" {
  fn linear_add(a: *const f32, b: *const f32, c: *mut f32, n: usize);
  fn matrix_mul(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    n: usize,
    m: usize,
    r: usize,
  );
}

#[derive(Debug)]
pub struct Matrix {
  rows: usize,
  cols: usize,
  data: Vec<f32>,
}

impl Matrix {
  pub fn new(rows: usize, cols: usize) -> Self {
    Matrix {
      rows,
      cols,
      data: vec![0.0; rows * cols],
    }
  }

  pub fn at(&self, row: usize, col: usize) -> &f32 {
    &self.data[row * self.cols + col]
  }
  pub fn at_mut(&mut self, row: usize, col: usize) -> &mut f32 {
    &mut self.data[row * self.cols + col]
  }

  fn impl_mul(&self, other: &Matrix) -> Matrix {
    assert_eq!(self.cols, other.rows);
    let mut result = Matrix::new(self.rows, other.cols);
    unsafe {
      matrix_mul(
        self.data.as_ptr(),
        other.data.as_ptr(),
        result.data.as_mut_ptr(),
        self.rows,
        self.cols,
        other.cols,
      );
    }
    result
  }
  fn impl_add(&self, other: &Matrix) -> Matrix {
    assert_eq!(self.rows, other.rows);
    assert_eq!(self.cols, other.cols);
    let mut result = Matrix::new(self.rows, self.cols);
    unsafe {
      linear_add(
        self.data.as_ptr(),
        other.data.as_ptr(),
        result.data.as_mut_ptr(),
        self.rows * self.cols,
      );
    }
    result
  }
}

impl Index<(usize, usize)> for Matrix {
  type Output = f32;
  fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
    self.at(row, col)
  }
}
impl IndexMut<(usize, usize)> for Matrix {
  fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
    self.at_mut(row, col)
  }
}
impl Mul for Matrix {
  type Output = Self;
  fn mul(self, other: Self) -> Self {
    self.impl_mul(&other)
  }
}
impl Add for Matrix {
  type Output = Matrix;

  fn add(self, rhs: Self) -> Self::Output {
    self.impl_add(&rhs)
  }
}
