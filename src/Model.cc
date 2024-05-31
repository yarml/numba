#include <Model.hh>
#include <cassert>
#include <string>

using namespace std;

pair<int, int> Model::inputShape() const {
  auto w0Shape = weights.front().shape();
  return {w0Shape.second, 1};
}
pair<int, int> Model::outputShape() const {
  auto wNShape = weights.back().shape();
  return {wNShape.first, 1};
}
vector<size_t> Model::layerSizes() const {
  vector<size_t> sizes;
  if (weights.empty()) {
    return move(sizes);
  }
  sizes.reserve(weights.size());
  sizes.push_back(weights.front().shape().second);
  for (auto &w : weights) {
    sizes.push_back(w.shape().first);
  }
  return move(sizes);
}

void Model::addLayer(NumbaMatrix weights, NumbaMatrix biases) {
  if (!this->weights.empty()) {
    auto prevWeightsShape = this->weights.back().shape();
    assert(weights.shape().second == prevWeightsShape.first);
  }
  assert(biases.shape().first == weights.shape().first &&
         biases.shape().second == 1);
  this->weights.push_back(move(weights));
  this->biases.push_back(move(biases));
}

NumbaMatrix Model::evaluate(NumbaMatrix input) const {
  assert(input.shape() == inputShape());
  NumbaMatrix currentLayer = move(input);
  for (size_t i = 0; i < weights.size(); ++i) {
    auto &w = weights[i];
    auto &b = biases[i];
    auto nextLayer = w * currentLayer + b;
    currentLayer = move(nextLayer);
  }
  assert(currentLayer.shape() == outputShape());
  return move(currentLayer);
}
