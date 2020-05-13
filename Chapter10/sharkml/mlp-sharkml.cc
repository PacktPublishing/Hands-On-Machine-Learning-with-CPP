#include "../data/data.h"

#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/LinearModel.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/Regularizer.h>

using namespace shark;

int main() {
  size_t n = 10000;
  size_t seed = 45345;
  auto data = GenerateData(-1.5, 1.5, n, seed, false);
  std::vector<RealVector> x_data(n);
  std::vector<RealVector> y_data(n);
  for (size_t i = 0; i < n; ++i) {
    x_data[i] = RealVector(1, data.first[i]);
    y_data[i] = RealVector(1, data.second[i]);
  }

  Data<RealVector> x = createDataFromRange(x_data);
  Data<RealVector> y = createDataFromRange(y_data);
  RegressionDataset train_data(x, y);

  using DenseLayer = LinearModel<RealVector, TanhNeuron>;

  DenseLayer layer1(1, 32, true);
  DenseLayer layer2(32, 16, true);
  DenseLayer layer3(16, 8, true);

  LinearModel<RealVector> output(8, 1, true);
  auto network = layer1 >> layer2 >> layer3 >> output;

  SquaredLoss<> loss;
  ErrorFunction<> error(train_data, &network, &loss, true);
  TwoNormRegularizer<> regularizer(error.numberOfVariables());
  double weight_decay = 0.0001;
  error.setRegularizer(weight_decay, &regularizer);
  error.init();

  initRandomNormal(network, 0.001);

  SteepestDescent<> optimizer;
  optimizer.setMomentum(0.5);
  optimizer.setLearningRate(0.01);
  optimizer.init(error);

  size_t epochs = 1000;
  size_t iterations = train_data.numberOfBatches();
  for (size_t epoch = 0; epoch != epochs; ++epoch) {
    double avg_loss = 0.0;
    for (size_t i = 0; i != iterations; ++i) {
      optimizer.step(error);
      if (i % 100 == 0) {
        avg_loss += optimizer.solution().value;
      }
    }
    avg_loss /= iterations;
    std::cout << "Epoch " << epoch << " | Avg. Loss " << avg_loss << std::endl;
  }
  network.setParameterVector(optimizer.solution().point);

  return 0;
}
