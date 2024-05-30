#include <numba.hh>
#include <raylib.h>
#include <iostream>
#include <sstream>

using namespace std;

int main() {
  auto labeled_images = loadLabeledImages("MNIST/train-images.idx3-ubyte",
                                          "MNIST/train-labels.idx1-ubyte");
  cout << "Loaded " << labeled_images.size() << " images" << endl;
  for (auto &image : labeled_images) {
    stringstream ss;
    ss << "Label: " << image.first;
    InitWindow(280, 280, ss.str().c_str());
    while (!WindowShouldClose()) {
      BeginDrawing();
      ClearBackground(RAYWHITE);
      for (size_t i = 0; i < 28; ++i) {
        for (size_t j = 0; j < 28; ++j) {
          DrawPixel(i, j,
                    Color{(unsigned char)(255 * image.second(i, j)),
                          (unsigned char)(255 * image.second(i, j)),
                          (unsigned char)(255 * image.second(i, j)), 255});
        }
      }
      EndDrawing();
    }
    CloseWindow();
  }
}