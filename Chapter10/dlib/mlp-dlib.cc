#include "../data/data.h"

#include <dlib/dnn.h>
#include <dlib/matrix.h>

#include <iostream>
#include <random>

int main() {
  using namespace dlib;

  size_t n = 10000;
  size_t seed = 45345;
  auto data = GenerateData(-1.5, 1.5, n, seed, false);

  std::vector<matrix<double>> x(n);
  std::vector<matrix<double>> y_data(n);

  for (size_t i = 0; i < n; ++i) {
    x[i].set_size(1, 1);
    x[i](0, 0) = data.first[i];

    y_data[i].set_size(1, 1);
    y_data[i](0, 0) = data.second[i];
  }

  // normalize data
  vector_normalizer<matrix<double>> normalizer_x;
  vector_normalizer<matrix<double>> normalizer_y;

  // let the normalizer learn the mean and standard deviation of the samples
  normalizer_x.train(x);
  normalizer_y.train(y_data);

  std::vector<float> y(n);

  // now normalize each sample
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = normalizer_x(x[i]);
    y_data[i] = normalizer_y(y_data[i]);
    y[i] = static_cast<float>(y_data[i](0, 0));
  }

  using NetworkType = loss_mean_squared<
      fc<1, htan<fc<8, htan<fc<16, htan<fc<32, input<matrix<double>>>>>>>>>>;
  NetworkType network;
  float weight_decay = 0.0001f;
  float momentum = 0.5f;
  sgd solver(weight_decay, momentum);
  dnn_trainer<NetworkType> trainer(network, solver);
  trainer.set_learning_rate(0.01);
  trainer.set_learning_rate_shrink_factor(1);  // disable learning rate changes
  trainer.set_mini_batch_size(64);
  trainer.set_max_num_epochs(500);
  trainer.be_verbose();
  trainer.train(x, y);
  network.clean();

  // auto predictions = network(new_x);

  return 0;
}
