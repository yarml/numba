mod cuda;
mod matrix;
mod model;
mod numba;

use matrix::Matrix;
use model::Model;
use raylib::{color::Color, drawing::RaylibDraw, ffi::KeyboardKey};

fn main() {
  let labeled_images = numba::load_labeled_images(
    "MNIST/train-images.idx3-ubyte".into(),
    "MNIST/train-labels.idx1-ubyte".into(),
  );

  let mut model = Model::new();
  model.add_layer(Matrix::random(16, 784), Matrix::random(16, 1));
  model.add_layer(Matrix::random(16, 16), Matrix::random(16, 1));
  model.add_layer(Matrix::random(10, 16), Matrix::random(10, 1));

  let test_case = labeled_images[0].2.to_linearized();
  let expectation = labeled_images[0].1.clone();

  let mse_before = model.evaluate_one(test_case.clone(), expectation.clone());
  let initial_result = model.apply(test_case.clone()).to_host().max_idx();

  for i in 0..100 {
    let upgrade = model.backdrop_once(test_case.clone(), expectation.clone());
    // println!("Phase {i}");
    // let upgrade_in_host: (Vec<_>, Vec<_>) = (
    //   upgrade.0.iter().map(|x| x.to_host()).collect(),
    //   upgrade.1.iter().map(|x| x.to_host()).collect(),
    // );
    // println!("{upgrade_in_host:?}");
    model = model.upgrade(&upgrade.0, &upgrade.1, 0.1);
  }
  let mse_after = model.evaluate_one(test_case.clone(), expectation.clone());

  let result = model.apply(test_case.clone()).to_host().max_idx();

  println!("MSE before: {mse_before}, MSE after: {mse_after}, initiali result: {initial_result}, final result: {result}");
}

fn browse_images(labeled_images: &[(usize, Matrix)]) {
  let (mut rl, thread) = raylib::init().size(400, 400).title("Test").build();
  let mut index = 0;
  while !rl.window_should_close() {
    if rl.is_key_pressed(KeyboardKey::KEY_LEFT) {
      index = (index + labeled_images.len() - 1) % labeled_images.len();
    }
    if rl.is_key_pressed(KeyboardKey::KEY_RIGHT) {
      index = (index + 1) % labeled_images.len();
    }
    let (label, image) = &labeled_images[index];
    let mut d = rl.begin_drawing(&thread);
    d.clear_background(Color::BLACK);
    d.draw_text(&format!("{label}"), 40, 40, 18, Color::WHITE);
    let image_in_host = image.to_host();
    for i in 0..image.shape().0 {
      for j in 0..image.shape().1 {
        let pixel = 255.0 * image_in_host[(i, j)];
        let color = Color::new(pixel as u8, pixel as u8, pixel as u8, 255);
        d.draw_pixel(i as i32, j as i32, color);
      }
    }
  }
}
