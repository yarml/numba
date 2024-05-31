use crate::matrix::Matrix;

pub struct Model {
  weights: Vec<Matrix>,
  biases: Vec<Matrix>,
}

impl Model {
  pub fn new() -> Self {
    Model {
      weights: Vec::new(),
      biases: Vec::new(),
    }
  }

  pub fn input_shape(&self) -> (usize, usize) {
    assert_ne!(self.weights.len(), 0);
    let w0_shape = self.weights[0].shape();
    (w0_shape.1, 1)
  }
  pub fn output_shape(&self) -> (usize, usize) {
    assert_ne!(self.weights.len(), 0);
    let wn_shape = self.weights[self.weights.len() - 1].shape();
    (wn_shape.0, 1)
  }
  pub fn layer_sizes(&self) -> Vec<usize> {
    let mut sizes = Vec::new();
    if self.weights.len() == 0 {
      return sizes;
    }
    sizes.reserve(self.weights.len() + 1);
    sizes.push(self.input_shape().0);
    for w in &self.weights {
      sizes.push(w.shape().0);
    }
    sizes
  }

  pub fn add_layer(&mut self, weights: Matrix, biases: Matrix) {
    if !self.weights.is_empty() {
      let prev_shape = self.weights[self.weights.len() - 1].shape();
      assert_eq!(weights.shape().1, prev_shape.0);
    }
    assert_eq!(biases.shape(), (weights.shape().0, 1));
    self.weights.push(weights);
    self.biases.push(biases);
  }
  pub fn evaluate(self, input: Matrix) -> Matrix {
    assert_eq!(input.shape(), self.input_shape());
    let mut current_layer = input;
    for (w, b) in self.weights.into_iter().zip(self.biases.into_iter()) {
      current_layer = (w * current_layer + b).to_sigmoided();
    }
    current_layer
  }
}
