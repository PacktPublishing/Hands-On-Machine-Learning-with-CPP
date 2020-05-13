#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/Models/LinearModel.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>

using namespace shark;

std::pair<Data<RealVector>, Data<RealVector>> GenerateData(size_t n) {
  std::vector<RealVector> x_data(n);
  std::vector<RealVector> y_data(n);

  std::random_device rd;
  std::mt19937 re(rd());
  std::uniform_real_distribution<double> dist(-1.5, 1.5);

  // generate data
  RealVector x_v(1);  // it's a typdef to remora::vector<float>
  RealVector y_v(1);
  for (size_t i = 0; i < n; ++i) {
    x_v(0) = i + dist(re);
    x_data[i] = x_v;

    y_v(0) = i + dist(re);       // add noise
    y_v(0) = 4. + 0.3 * y_v(0);  // line coeficients
    y_data[i] = y_v;
  }

  return {createDataFromRange(x_data), createDataFromRange(y_data)};
}

int main() {
  Data<RealVector> x;
  Data<RealVector> y;
  std::tie(x, y) = GenerateData(1000);
  RegressionDataset data(x, y);
  LinearModel<> model;
  LinearRegression trainer;
  trainer.train(model, data);

  // We can get calculated parameters vector
  auto b = model.parameterVector();
  std::cout << "Estimated parameters :\n" << b << std::endl;

  // Also we can calculate value of Squared Error
  SquaredLoss<> loss;
  Data<RealVector> prediction = model(x);
  auto se = loss(y, prediction);
  std::cout << "Squared Error :\n" << se << std::endl;

  // For new X data you can predict new Y
  std::vector<RealVector> new_x_data;
  new_x_data.push_back({1});
  new_x_data.push_back({2});
  new_x_data.push_back({3});
  prediction = model(createDataFromRange(new_x_data));
  std::cout << "Predictions \n" << prediction << std::endl;

  return 0;
}
