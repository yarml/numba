use std::ops::{Add, Index, IndexMut, Mul, Sub};

use rand::Rng;

use crate::cuda::CudaBox;

extern "C" {
  fn numba_VecAdd(
    dev_a: *const f32,
    dev_b: *const f32,
    dev_c: *const f32,
    n: usize,
  );
  fn numba_VecSub(
    dev_a: *const f32,
    dev_b: *const f32,
    dev_c: *const f32,
    n: usize,
  );
  fn numba_VecMul(
    dev_a: *const f32,
    dev_b: *const f32,
    dev_c: *const f32,
    n: usize,
  );
  fn numba_MatrixMul(
    dev_a: *const f32,
    dev_b: *const f32,
    dev_c: *const f32,
    n: usize,
    m: usize,
    r: usize,
  );
  fn numba_VecSigmoid(dev_a: *const f32, dev_b: *const f32, n: usize);
  fn numba_VecSigmoidPrime(dev_a: *const f32, dev_b: *const f32, n: usize);
  fn numba_VecDot(dev_a: *const f32, dev_b: *const f32, n: usize) -> f32;
  fn numba_MatrixTranspose(
    dev_a: *const f32,
    dev_b: *const f32,
    n: usize,
    m: usize,
  );
  fn numba_VecScale(dev_a: *const f32, dev_b: *const f32, s: f32, n: usize);
}

#[derive(Clone)]
pub struct Matrix {
  rows: usize,
  cols: usize,
  data: CudaBox<[f32]>,
}

impl Matrix {
  pub fn new(rows: usize, cols: usize) -> Self {
    Matrix {
      rows,
      cols,
      data: CudaBox::new_zeroed_slice(rows * cols),
    }
  }
  pub fn random(rows: usize, cols: usize) -> Self {
    let mut rng = rand::thread_rng();
    let data: CudaBox<[f32]> =
      (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
    Matrix { rows, cols, data }
  }

  pub fn to_host(&self) -> MatrixInHost {
    let data = self.data.to_host_slice();
    MatrixInHost {
      rows: self.rows,
      cols: self.cols,
      data,
    }
  }
  pub fn transpose(&self) -> Matrix {
    let result = Matrix::new(self.cols, self.rows);
    unsafe {
      numba_MatrixTranspose(
        self.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
        self.rows,
        self.cols,
      );
    }
    result
  }

  pub fn shape(&self) -> (usize, usize) {
    (self.rows, self.cols)
  }

  pub fn to_linearized(&self) -> Matrix {
    let mut result = Matrix::new(self.rows * self.cols, 1);
    result.data = self.data.clone();
    result
  }

  pub fn sigmoid(&self) -> Matrix {
    let result = Matrix::new(self.rows, self.cols);
    unsafe {
      numba_VecSigmoid(
        self.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
        self.rows * self.cols,
      );
    }
    result
  }
  pub fn sigmoid_prime(&self) -> Matrix {
    let result = Matrix::new(self.rows, self.cols);
    unsafe {
      numba_VecSigmoidPrime(
        self.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
        self.rows * self.cols,
      );
    }
    result
  }
  pub fn dot(&self, other: &Matrix) -> f32 {
    assert_eq!(self.rows, other.rows);
    assert_eq!(self.cols, other.cols);
    unsafe {
      numba_VecDot(
        self.data.as_ptr() as *const f32,
        other.data.as_ptr() as *const f32,
        self.rows * self.cols,
      )
    }
  }
  pub fn scale(&self, s: f32) -> Matrix {
    let result = Matrix::new(self.rows, self.cols);
    unsafe {
      numba_VecScale(
        self.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
        s,
        self.rows * self.cols,
      );
    }
    result
  }

  pub fn mse(&self, other: &Matrix) -> f32 {
    assert_eq!(self.rows, other.rows);
    assert_eq!(self.cols, other.cols);
    let diff = self - other;
    diff.dot(&diff) * (1.0 / (self.rows * self.cols) as f32)
  }
  pub fn element_mul(&self, other: &Matrix) -> Matrix {
    assert_eq!(self.rows, other.rows);
    assert_eq!(self.cols, other.cols);
    let result = Matrix::new(self.rows, self.cols);
    unsafe {
      numba_VecMul(
        self.data.as_ptr() as *const f32,
        other.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
        self.rows * self.cols,
      );
    }
    result
  }

  fn impl_mul(&self, other: &Matrix) -> Matrix {
    assert_eq!(self.cols, other.rows);
    let result = Matrix::new(self.rows, other.cols);
    unsafe {
      numba_MatrixMul(
        self.data.as_ptr() as *const f32,
        other.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
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
    let result = Matrix::new(self.rows, self.cols);
    unsafe {
      numba_VecAdd(
        self.data.as_ptr() as *const f32,
        other.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
        self.rows * self.cols,
      );
    }
    result
  }
  fn impl_sub(&self, other: &Matrix) -> Matrix {
    assert_eq!(self.rows, other.rows);
    assert_eq!(self.cols, other.cols);
    let result = Matrix::new(self.rows, self.cols);
    unsafe {
      numba_VecSub(
        self.data.as_ptr() as *const f32,
        other.data.as_ptr() as *const f32,
        result.data.as_ptr() as *const f32,
        self.rows * self.cols,
      );
    }
    result
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
impl Sub for Matrix {
  type Output = Matrix;

  fn sub(self, rhs: Self) -> Self::Output {
    self.impl_sub(&rhs)
  }
}
impl Mul for &Matrix {
  type Output = Matrix;
  fn mul(self, other: Self) -> Self::Output {
    self.impl_mul(other)
  }
}
impl Add for &Matrix {
  type Output = Matrix;

  fn add(self, rhs: Self) -> Self::Output {
    self.impl_add(rhs)
  }
}
impl Sub for &Matrix {
  type Output = Matrix;

  fn sub(self, rhs: Self) -> Self::Output {
    self.impl_sub(rhs)
  }
}

#[derive(Debug, Clone)]
pub struct MatrixInHost {
  rows: usize,
  cols: usize,
  data: Box<[f32]>,
}

impl MatrixInHost {
  pub fn new(rows: usize, cols: usize) -> Self {
    MatrixInHost {
      rows,
      cols,
      data: vec![0.0; rows * cols].into_boxed_slice(),
    }
  }
  pub fn to_device(&self) -> Matrix {
    let data: CudaBox<[f32]> = CudaBox::new_zeroed_slice(self.rows * self.cols);
    data.to_device(self.data.clone());
    Matrix {
      rows: self.rows,
      cols: self.cols,
      data,
    }
  }
  pub fn at(&self, row: usize, col: usize) -> &f32 {
    &self.data[row * self.cols + col]
  }
  pub fn at_mut(&mut self, row: usize, col: usize) -> &mut f32 {
    &mut self.data[row * self.cols + col]
  }

  pub fn max(&self) -> f32 {
    self
      .data
      .iter()
      .map(|&x| x)
      .fold(f32::NEG_INFINITY, f32::max)
  }
  pub fn max_idx(&self) -> usize {
    fn max(a: (usize, f32), b: (usize, f32)) -> (usize, f32) {
      if a.1 > b.1 {
        a
      } else {
        b
      }
    }
    self
      .data
      .iter()
      .map(|&x| x)
      .enumerate()
      .fold((usize::MAX, f32::NEG_INFINITY), max)
      .0
  }
}

impl Index<(usize, usize)> for MatrixInHost {
  type Output = f32;
  fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
    self.at(row, col)
  }
}
impl IndexMut<(usize, usize)> for MatrixInHost {
  fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
    self.at_mut(row, col)
  }
}
