use std::fmt::format;

use raylib::{
  color::Color,
  drawing::RaylibDraw,
  ffi::{CloseWindow, KeyboardKey},
};

mod matrix;
mod numba;

fn main() {
  let labeled_images = numba::load_labeled_images(
    "MNIST/train-images.idx3-ubyte".into(),
    "MNIST/train-labels.idx1-ubyte".into(),
  );

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
    for i in 0..image.shape().0 {
      for j in 0..image.shape().1 {
        let pixel = 255.0 * image[(i, j)];
        let color = Color::new(pixel as u8, pixel as u8, pixel as u8, 255);
        d.draw_pixel(i as i32, j as i32, color);
      }
    }
  }
}
