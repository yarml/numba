mod cuda;
mod matrix;
mod model;
mod numba;

use std::{
  io::{self, Write},
  time::Instant,
};

use matrix::Matrix;
use model::Model;
use raylib::{color::Color, drawing::RaylibDraw, ffi::KeyboardKey};

fn main() {
  let labeled_images = numba::load_labeled_images(
    "MNIST/train-images.idx3-ubyte".into(),
    "MNIST/train-labels.idx1-ubyte".into(),
  );

  let (data_with_expectations, data_with_labels): (Vec<_>, Vec<_>) =
    labeled_images
      .into_iter()
      .map(|(label, expectation, image)| {
        (
          (expectation, image.to_linearized()),
          (label, image.to_linearized()),
        )
      })
      .unzip();

  let mut model = Model::new();
  model.add_layer(Matrix::random(16, 784), Matrix::random(16, 1));
  model.add_layer(Matrix::random(16, 16), Matrix::random(16, 1));
  model.add_layer(Matrix::random(10, 16), Matrix::random(10, 1));

  println!("Starting initial evaluation");

  let init_eval_start = Instant::now();
  let correct_guesses = model.evaluate_all(&data_with_labels);
  let init_eval_end = Instant::now();

  println!(
    "Before training, the model guessed {}/{} images correctly ({}%) in {}ms",
    correct_guesses,
    data_with_labels.len(),
    100.0 * correct_guesses as f32 / data_with_labels.len() as f32,
    init_eval_end.duration_since(init_eval_start).as_millis()
  );

  for n in 0..10 {
    print!("Phase {n}...");
    io::stdout().flush().unwrap();
    let phase_start = Instant::now();
    model.train_once_by_batch(&data_with_expectations, 100, 0.1);
    let phase_end = Instant::now();
    println!(
      " ({}ms)",
      phase_end.duration_since(phase_start).as_millis()
    );
  }

  println!("Training done!");
  println!("Starting final evaluation");
  let correct_guesses = model.evaluate_all(&data_with_labels);
  println!(
    "After training, the model guessed {}/{} images correctly ({}%)",
    correct_guesses,
    data_with_labels.len(),
    100.0 * correct_guesses as f32 / data_with_labels.len() as f32
  );
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
