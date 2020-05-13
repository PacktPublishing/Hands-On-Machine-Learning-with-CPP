#include "../data/data.h"

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/neuralnets/NeuralLayers.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/util/factory.h>
#include <iostream>
#include <random>

using namespace shogun;

int main(int, char*[]) {
  shogun::init_shogun_with_defaults();
  shogun::sg_io->set_loglevel(shogun::MSG_DEBUG);

  size_t n = 10000;
  size_t seed = 45345;
  auto data = GenerateData(-1.5, 1.5, n, seed, false);

  SGMatrix<float64_t> x_values(1, static_cast<index_t>(n));
  SGVector<float64_t> y_values(static_cast<index_t>(n));

  for (size_t i = 0; i < n; ++i) {
    x_values.set_element(data.first[i], 0, static_cast<index_t>(i));
    y_values.set_element(data.second[i], static_cast<index_t>(i));
  }

  auto x = some<CDenseFeatures<float64_t>>(x_values);
  auto y = some<CRegressionLabels>(y_values);

  auto dimensions = x->get_num_features();
  auto layers = some<CNeuralLayers>();
  layers = wrap(layers->input(dimensions));
  layers = wrap(layers->rectified_linear(32));
  layers = wrap(layers->rectified_linear(16));
  layers = wrap(layers->rectified_linear(8));
  layers = wrap(layers->linear(1));
  auto all_layers = layers->done();

  auto network = some<CNeuralNetwork>(all_layers);
  network->quick_connect();
  network->initialize_neural_network();

  network->set_optimization_method(NNOM_GRADIENT_DESCENT);
  network->set_gd_mini_batch_size(64);
  network->set_l2_coefficient(0.0001);  // regularization
  network->set_max_num_epochs(500);
  network->set_epsilon(0.0);  // convergence criteria
  network->set_gd_learning_rate(0.01);
  network->set_gd_momentum(0.5);

  network->set_labels(y);
  // loss function is not configurable
  network->train(x);

  auto labels_predict = network->apply_regression(x);
  auto err = some<CMeanSquaredError>();
  auto mse = err->evaluate(labels_predict, y);

  std::cout << "Total Loss " << mse << std::endl;

  shogun::exit_shogun();
  return 0;
}
