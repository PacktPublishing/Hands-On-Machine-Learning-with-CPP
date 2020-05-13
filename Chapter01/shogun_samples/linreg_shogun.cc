#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/util/factory.h>
#include <iostream>
#include <random>

using namespace shogun;

int main(int, char*[]) {
  shogun::init_shogun_with_defaults();
  shogun::sg_io->set_loglevel(shogun::MSG_DEBUG);

  const int32_t n = 1000;
  SGMatrix<float64_t> x_values(1, n);
  SGVector<float64_t> y_values(n);

  std::random_device rd;
  std::mt19937 re(rd());
  std::uniform_real_distribution<double> dist(-1.5, 1.5);

  // generate data
  for (int32_t i = 0; i < n; ++i) {
    x_values.set_element(i + dist(re), 0, i);

    auto y_val = i + dist(re);  // add noise
    y_val = 4. + 0.3 * y_val;   // line coeficients
    y_values.set_element(y_val, i);
  }

  auto x = some<CDenseFeatures<float64_t>>(x_values);
  auto y = some<CRegressionLabels>(y_values);

  float64_t tau_regularization = 0.0001;
  auto lr = some<CLinearRidgeRegression>(tau_regularization, nullptr, nullptr);
  lr->set_labels(y);
  if (!lr->train(x)) {
    std::cerr << "training failed\n";
  }

  // e can get calculated bias term
  auto bias = lr->get_bias();
  std::cout << "Bias = " << bias << "\n";

  // We can get calculated parameters vector
  auto weights = lr->get_w();
  std::cout << "Weights = \n";
  for (int32_t i = 0; i < weights.size(); ++i) {
    std::cout << weights[i] << "\n";
  }

  // Also we can calculate value of Mean Squared Error:
  auto y_predict = lr->apply_regression(x);
  auto eval = some<CMeanSquaredError>();
  auto mse = eval->evaluate(y_predict, y);
  std::cout << "MSE = " << mse << std::endl;

  //  For new X data you can predict new Y
  SGMatrix<float64_t> new_x_values(1, 5);
  new_x_values.set_element(1, 0, 0);
  new_x_values.set_element(2, 0, 1);
  new_x_values.set_element(3, 0, 2);
  new_x_values.set_element(4, 0, 3);
  new_x_values.set_element(5, 0, 4);
  auto new_x = some<CDenseFeatures<float64_t>>(new_x_values);
  y_predict = lr->apply_regression(new_x);
  std::cout << "Predicted values\n" << y_predict->to_string() << std::endl;

  shogun::exit_shogun();
  return 0;
}
