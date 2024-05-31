use matrix::Matrix;

mod matrix;


fn main() {
  let mut a = Matrix::new(2, 1);
  a[(0, 0)] = 1.0;
  a[(1, 0)] = 3.0;

  let mut b = Matrix::new(1, 2);
  b[(0, 0)] = 5.0;
  b[(0, 1)] = 6.0;

  let c = a * b;

  println!("{:?}", c);
}
