use std::{
  fs::File,
  io::{BufReader, Read},
  path::PathBuf,
};

use crate::matrix::Matrix;

pub fn load_labeled_images(
  images_path: PathBuf,
  labels_path: PathBuf,
) -> Vec<(usize, Matrix)> {
  let mut images = BufReader::new(File::open(images_path).unwrap());
  let mut labels = BufReader::new(File::open(labels_path).unwrap());

  {
    let mut images_magic = [0u8; 4];
    let mut labels_magic = [0u8; 4];
    images.read_exact(&mut images_magic).unwrap();
    labels.read_exact(&mut labels_magic).unwrap();
    let images_magic = i32::from_be_bytes(images_magic);
    let labels_magic = i32::from_be_bytes(labels_magic);
    assert_eq!(images_magic, 0x803);
    assert_eq!(labels_magic, 0x801);
  }

  let images_count = {
    let mut images_count = [0u8; 4];
    let mut labels_count = [0u8; 4];
    images.read_exact(&mut images_count).unwrap();
    labels.read_exact(&mut labels_count).unwrap();
    let images_count = i32::from_be_bytes(images_count) as usize;
    let labels_count = i32::from_be_bytes(labels_count) as usize;
    assert_eq!(images_count, labels_count);
    images_count
  };

  let (rows, cols) = {
    let mut rows = [0u8; 4];
    let mut cols = [0u8; 4];
    images.read_exact(&mut rows).unwrap();
    images.read_exact(&mut cols).unwrap();
    let rows = i32::from_be_bytes(rows) as usize;
    let cols = i32::from_be_bytes(cols) as usize;
    (rows, cols)
  };

  let mut labeled_images = Vec::with_capacity(images_count);

  for _ in 0..images_count {
    let mut pixels = vec![0u8; rows * cols];
    images.read_exact(&mut pixels).unwrap();
    let label = {
      let mut label = [0u8; 1];
      labels.read_exact(&mut label).unwrap();
      label[0] as usize
    };
    let mut matrix = Matrix::new(rows, cols);
    for j in 0..cols {
      for i in 0..rows {
        matrix[(i, j)] = pixels[j * cols + i] as f32 / 255.0;
      }
    }
    labeled_images.push((label, matrix));
  }

  labeled_images
}
