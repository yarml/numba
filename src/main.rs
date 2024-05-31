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

  let output = model
    .evaluate(labeled_images[0].1.to_linearized())
    .to_host();
  println!("{:?}", output);
  println!("Most likely: {}", output.max_idx());
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
