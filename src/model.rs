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

  pub fn evaluate_all(&self, data: &[(usize, Matrix)]) -> usize {
    data
      .iter()
      .filter(|(label, input)| {
        let output = self.apply(input.clone()).to_host().max_idx();
        output == *label
      })
      .count()
  }

  // Returns ∇C(b, w)
  pub fn backdrop_once(
    &self,
    input: Matrix,
    expectation: Matrix,
  ) -> (Vec<Matrix>, Vec<Matrix>) {
    assert_eq!(expectation.shape(), self.output_shape());
    let mut nabla_biases: Vec<_> =
      self.biases.iter().map(|_| Matrix::new(0, 0)).collect();
    let mut nabla_weights: Vec<_> =
      self.weights.iter().map(|_| Matrix::new(0, 0)).collect();
    let (layers, full_layers) = self.impl_run_model(input);
    let cost_deriv = layers.last().unwrap() - &expectation;
    let mut delta =
      cost_deriv.element_mul(&layers.last().unwrap().sigmoid_prime());
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

  pub fn train_once_all(
    &mut self,
    training_data: &[(Matrix, Matrix)],
    rate: f32,
  ) {
    let mut nabla_biases: Vec<_> = self
      .biases
      .iter()
      .map(|b| Matrix::new(b.shape().0, b.shape().1))
      .collect();
    let mut nabla_weights: Vec<_> = self
      .weights
      .iter()
      .map(|w| Matrix::new(w.shape().0, w.shape().1))
      .collect();
    for (expectation, input) in training_data {
      let (nabla_biases_n, nabla_weights_n) =
        self.backdrop_once(input.clone(), expectation.clone());
      nabla_biases = nabla_biases
        .into_iter()
        .zip(nabla_biases_n.iter())
        .map(|(sum_nb, nbn)| &sum_nb + nbn)
        .collect();
      nabla_weights = nabla_weights
        .into_iter()
        .zip(nabla_weights_n.iter())
        .map(|(sum_nw, nwn)| &sum_nw + nwn)
        .collect();
    }
    nabla_biases = nabla_biases
      .into_iter()
      .map(|nb| nb.scale(1.0 / training_data.len() as f32))
      .collect();
    nabla_weights = nabla_weights
      .into_iter()
      .map(|nw| nw.scale(1.0 / training_data.len() as f32))
      .collect();
    self.upgrade(&nabla_biases, &nabla_weights, rate);
  }

  pub fn train_once_by_batch(
    &mut self,
    training_data: &[(Matrix, Matrix)],
    batch_size: usize,
    rate: f32,
  ) {
    assert_eq!(training_data.len() % batch_size, 0);
    for batch in training_data.chunks(batch_size) {
      self.train_once_all(batch, rate);
    }
  }

  pub fn upgrade(
    &mut self,
    nabla_biases: &[Matrix],
    nabla_weights: &[Matrix],
    rate: f32,
  ) {
    self.biases = self
      .biases
      .iter()
      .zip(nabla_biases.iter())
      .map(|(bias, nabla_bias)| bias - &nabla_bias.scale(rate))
      .collect();
    self.weights = self
      .weights
      .iter()
      .zip(nabla_weights.iter())
      .map(|(weight, nabla_weight)| weight - &nabla_weight.scale(rate))
      .collect();
  }
}
