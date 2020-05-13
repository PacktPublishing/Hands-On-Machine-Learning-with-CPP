#include <dlib/dnn.h>
#include <dlib/matrix.h>

#include <iostream>
#include <random>

using namespace dlib;

using NetworkType = loss_mean_squared<fc<1, input<matrix<double>>>>;
using SampleType = matrix<double, 1, 1>;
using KernelType = linear_kernel<SampleType>;

float func(float x) {
  return 4.f + 0.3f * x;  // line coeficients
}

void TrainAndSaveKRR(const std::vector<matrix<double>>& x,
                     const std::vector<float>& y) {
  krr_trainer<KernelType> trainer;
  trainer.set_kernel(KernelType());
  decision_function<KernelType> df = trainer.train(x, y);
  serialize("dlib-krr.dat") << df;
}

void LoadAndPredictKRR(const std::vector<matrix<double>>& x) {
  decision_function<KernelType> df;

  deserialize("dlib-krr.dat") >> df;

  // Predict

  std::cout << "KRR predictions \n";
  for (auto& v : x) {
    auto p = df(v);
    std::cout << static_cast<double>(p) << std::endl;
  }
}

void TrainAndSaveNetwork(const std::vector<matrix<double>>& x,
                         const std::vector<float>& y) {
  NetworkType network;
  sgd solver;
  dnn_trainer<NetworkType> trainer(network, solver);
  trainer.set_learning_rate(0.0001);
  trainer.set_mini_batch_size(50);
  trainer.set_max_num_epochs(300);
  trainer.be_verbose();
  trainer.train(x, y);
  network.clean();

  serialize("dlib-net.dat") << network;
  net_to_xml(network, "net.xml");
}

void LoadAndPredictNetwork(const std::vector<matrix<double>>& x) {
  NetworkType network;

  deserialize("dlib-net.dat") >> network;

  // Predict
  auto predictions = network(x);

  std::cout << "Net predictions \n";
  for (auto p : predictions) {
    std::cout << static_cast<double>(p) << std::endl;
  }
}

int main() {
  size_t n = 1000;
  std::vector<matrix<double>> x(n);
  std::vector<float> y(n);

  std::random_device rd;
  std::mt19937 re(rd());
  std::uniform_real_distribution<float> dist(-1.5, 1.5);

  // generate data
  for (size_t i = 0; i < n; ++i) {
    x[i].set_size(1, 1);
    x[i](0, 0) = i;

    y[i] = func(i) + dist(re);
  }

  // normalize data
  vector_normalizer<matrix<double>> normalizer_x;
  // let the normalizer learn the mean and standard deviation of the samples
  normalizer_x.train(x);
  // now normalize each sample
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = normalizer_x(x[i]);
  }

  TrainAndSaveNetwork(x, y);
  TrainAndSaveKRR(x, y);

  // Generate new data
  std::cout << "Target values \n";
  std::vector<matrix<double>> new_x(5);
  for (size_t i = 0; i < 5; ++i) {
    new_x[i].set_size(1, 1);
    new_x[i](0, 0) = i;
    new_x[i] = normalizer_x(new_x[i]);
    std::cout << func(i) << std::endl;
  }

  // Predict
  LoadAndPredictNetwork(new_x);
  LoadAndPredictKRR(new_x);

  return 0;
}
