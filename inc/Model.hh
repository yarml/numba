#include <Matrix.hh>
#include <cstddef>
#include <vector>

class Model {
private:
  std::vector<NumbaMatrix> weights;
  std::vector<NumbaMatrix> biases;

public:
  Model() = default;

public:
  Model &operator=(Model const &) = delete;
  Model(Model const &) = delete;

public:
  std::pair<int, int> inputShape() const;
  std::pair<int, int> outputShape() const;
  std::vector<size_t> layerSizes() const;

public:
  NumbaMatrix evaluate(NumbaMatrix input) const;
  void addLayer(NumbaMatrix weights, NumbaMatrix biases);
};
