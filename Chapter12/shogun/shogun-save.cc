#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>

#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/io/SerializableHdf5File.h>
#include <shogun/io/SerializableJsonFile.h>
// #include <shogun/io/SerializableXmlFile.h>
#include <shogun/io/NeuralNetworkFileReader.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/neuralnets/NeuralLayers.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/util/factory.h>
#include <iostream>
#include <random>

using namespace shogun;

double func(double x) {
  return 4. + 0.3 * x;
}

void TrainAndSaveLRR(Some<CDenseFeatures<float64_t>> x,
                     Some<CRegressionLabels> y) {
  float64_t tau_regularization = 0.0001;
  auto model =
      some<CLinearRidgeRegression>(tau_regularization, nullptr, nullptr);
  model->set_labels(y);
  if (!model->train(x)) {
    std::cerr << "training failed\n";
  }

  auto file = some<CSerializableHdf5File>("shogun-lr.dat", 'w');
  if (!model->save_serializable(file)) {
    std::cerr << "Failed to save the model\n";
  }
}

void LoadAndPredictLRR(Some<CDenseFeatures<float64_t>> x) {
  auto file = some<CSerializableHdf5File>("shogun-lr.dat", 'r');
  auto model = some<CLinearRidgeRegression>();
  if (model->load_serializable(file)) {
    auto new_x = some<CDenseFeatures<float64_t>>(x);
    auto y_predict = model->apply_regression(new_x);
    std::cout << "LR predicted values: \n"
              << y_predict->to_string() << std::endl;
  }
}

void TrainAndSaveNET(Some<CDenseFeatures<float64_t>> x,
                     Some<CRegressionLabels> y) {
  auto dimensions = x->get_num_features();
  auto layers = some<CNeuralLayers>();
  layers = wrap(layers->input(dimensions));
  layers = wrap(layers->linear(1));
  auto all_layers = layers->done();

  auto network = some<CNeuralNetwork>(all_layers);
  network->quick_connect();
  network->initialize_neural_network();

  network->set_optimization_method(NNOM_GRADIENT_DESCENT);
  network->set_gd_mini_batch_size(0);
  network->set_max_num_epochs(1000);
  network->set_gd_learning_rate(0.01);
  network->set_gd_momentum(0.9);

  network->set_labels(y);
  if (network->train(x)) {
    auto file = some<CSerializableHdf5File>("shogun-net.dat", 'w');
    if (!network->save_serializable(file)) {
      std::cerr << "Failed to save the model\n";
    }
  } else {
    std::cerr << "Failed to train the network\n";
  }
}

Some<CNeuralNetwork> NETFromJson() {
  CNeuralNetworkFileReader reader;
  const char* net_str =
      "{"
      "	\"optimization_method\": \"NNOM_GRADIENT_DESCENT\","
      "	\"max_num_epochs\": 1000,"
      "	\"gd_mini_batch_size\": 0,"
      "	\"gd_learning_rate\": 0.01,"
      "	\"gd_momentum\": 0.9,"

      "	\"layers\":"
      "	{"
      "		\"input1\":"
      "		{"
      "			\"type\": \"NeuralInputLayer\","
      "			\"num_neurons\": 1,"
      "			\"start_index\": 0"
      "		},"
      "		\"linear1\":"
      "		{"
      "			\"type\": \"NeuralLinearLayer\","
      "			\"num_neurons\": 1,"
      "			\"inputs\": [\"input1\"]"
      "		}"
      "	}"
      "}";
  auto network = wrap(reader.read_string(net_str));

  return network;
}

void LoadAndPredictNET(Some<CDenseFeatures<float64_t>> x) {
  auto file = some<CSerializableHdf5File>("shogun-net.dat", 'r');

  //  auto dimensions = 1;
  //  auto layers = some<CNeuralLayers>();
  //  layers = wrap(layers->input(dimensions));
  //  layers = wrap(layers->linear(1));
  //  auto all_layers = layers->done();

  //  auto network = some<CNeuralNetwork>(all_layers);
  //  network->quick_connect();

  auto network = NETFromJson();

  if (network->load_serializable(file)) {
    auto new_x = some<CDenseFeatures<float64_t>>(x);
    auto y_predict = network->apply_regression(new_x);
    std::cout << "Network predicted values: \n"
              << y_predict->to_string() << std::endl;
  }
}

int main(int, char*[]) {
  shogun::init_shogun_with_defaults();
  // shogun::sg_io->set_loglevel(shogun::MSG_DEBUG);

  const int32_t n = 1000;
  SGMatrix<float64_t> x_values(1, n);
  SGVector<float64_t> y_values(n);

  std::random_device rd;
  std::mt19937 re(rd());
  std::uniform_real_distribution<double> dist(-1.5, 1.5);

  // generate data
  for (int32_t i = 0; i < n; ++i) {
    x_values.set_element(i, 0, i);

    auto y_val = func(i) + dist(re);
    y_values.set_element(y_val, i);
  }

  auto x = some<CDenseFeatures<float64_t>>(x_values);
  auto y = some<CRegressionLabels>(y_values);

  // rescale
  auto x_scaler = some<CRescaleFeatures>();
  x_scaler->fit(x);
  x_scaler->transform(x, true);

  SGMatrix<float64_t> new_x_values(1, 5);
  std::cout << "Target values : \n";
  for (index_t i = 0; i < 5; ++i) {
    new_x_values.set_element(static_cast<double>(i), 0, i);
    std::cout << func(i) << std::endl;
  }

  auto new_x = some<CDenseFeatures<float64_t>>(new_x_values);
  x_scaler->transform(new_x, true);

  TrainAndSaveLRR(x, y);
  LoadAndPredictLRR(new_x);

  TrainAndSaveNET(x, y);
  LoadAndPredictNET(new_x);

  shogun::exit_shogun();
  return 0;
}
