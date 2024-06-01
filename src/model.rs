use std::vec;

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

  // Returns the state of all layers, the state of all layers before the sigmoid
  fn impl_run_model(&self, input: Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
    assert_eq!(input.shape(), self.input_shape());
    let mut layers = vec![input.sigmoid()];
    let mut full_layers = vec![input];
    for (w, b) in self.weights.iter().zip(self.biases.iter()) {
      let current_layer = layers.last().unwrap();
      let next_layer = &(w * current_layer) + b;
      layers.push(next_layer.sigmoid());
      full_layers.push(next_layer);
    }
    (layers, full_layers)
  }

  pub fn apply(&self, input: Matrix) -> Matrix {
    let (mut layers, _) = self.impl_run_model(input);
    layers.pop().unwrap()
  }

  pub fn evaluate_one(&self, input: Matrix, expectation: Matrix) -> f32 {
    assert_eq!(expectation.shape(), self.output_shape());
    let (layers, _) = self.impl_run_model(input);
    let output = layers.last().unwrap();
    output.mse(&expectation)
  }

  // Returns ∇C(b, w)
  pub fn backdrop_once(
    &self,
    input: Matrix,
    expectation: Matrix,
  ) -> (Vec<Matrix>, Vec<Matrix>) {
    assert_eq!(expectation.shape(), self.output_shape());
    let mut nabla_biases: Vec<_> = self
      .biases
      .iter()
      .map(|bias| Matrix::new(bias.shape().0, bias.shape().1))
      .collect();
    let mut nabla_weights: Vec<_> = self
      .weights
      .iter()
      .map(|weight| Matrix::new(weight.shape().0, weight.shape().1))
      .collect();
    let (layers, full_layers) = self.impl_run_model(input);
    let cost_deriv = layers.last().unwrap() - &expectation;
    let mut delta = cost_deriv
      .element_mul(&layers.last().unwrap().sigmoid_prime())
      .scale(2.0);
    *nabla_biases.last_mut().unwrap() = delta.clone();
    *nabla_weights.last_mut().unwrap() =
      &delta * &layers[layers.len() - 2].transpose();
    for l in 2..self.layer_sizes().len() {
      let current_full_layer = &full_layers[full_layers.len() - l];
      let prime = current_full_layer.sigmoid_prime();
      delta = (&self.weights[self.weights.len() - l + 1].transpose() * &delta)
        .element_mul(&prime);
      let idx = nabla_biases.len() - l;
      nabla_biases[idx] = delta.clone();
      nabla_weights[idx] = &delta * &layers[layers.len() - l - 1].transpose();
    }
    (nabla_biases, nabla_weights)
  }

  pub fn upgrade(
    &self,
    nabla_biases: &[Matrix],
    nabla_weights: &[Matrix],
    rate: f32,
  ) -> Self {
    let new_biases: Vec<_> = self
      .biases
      .iter()
      .zip(nabla_biases.iter())
      .map(|(bias, nabla_bias)| bias - &nabla_bias.scale(rate))
      .collect();
    let new_weights: Vec<_> = self
      .weights
      .iter()
      .zip(nabla_weights.iter())
      .map(|(weight, nabla_weight)| weight - &nabla_weight.scale(rate))
      .collect();
    Model {
      weights: new_weights,
      biases: new_biases,
    }
  }
}
