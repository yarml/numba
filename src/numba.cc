#include <Model.hh>
#include <iostream>
#include <numba.hh>

using namespace std;

int main() {
  auto labeledImages = loadLabeledImages("MNIST/train-images.idx3-ubyte",
                                         "MNIST/train-labels.idx1-ubyte");
  Model model;
  model.addLayer(NumbaMatrix(16, 28 * 28), NumbaMatrix(16, 1));
  model.addLayer(NumbaMatrix(16, 16), NumbaMatrix(16, 1));
  model.addLayer(NumbaMatrix(10, 16), NumbaMatrix(10, 1));

  for (auto size : model.layerSizes()) {
    cout << size << endl;
  }
  auto input = labeledImages[0].second.linearize();
  auto output = model.evaluate(move(input));
  cout << output.shape().first << ' ' << output.shape().second << endl;
}