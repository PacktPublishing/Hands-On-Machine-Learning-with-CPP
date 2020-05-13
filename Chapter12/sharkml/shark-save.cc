#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/Models/LinearModel.h>
#include <boost/archive/polymorphic_binary_iarchive.hpp>
#include <boost/archive/polymorphic_binary_oarchive.hpp>
using namespace shark;

double func(double x) {
  return 4. + 0.3 * x;  // line coeficients
}

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
    x_v(0) = i;
    x_data[i] = x_v;

    y_v(0) = func(i) + dist(re);  // add noise
    y_data[i] = y_v;
  }

  return {createDataFromRange(x_data), createDataFromRange(y_data)};
}

int main() {
  {
    Data<RealVector> x;
    Data<RealVector> y;
    std::tie(x, y) = GenerateData(1000);
    RegressionDataset data(x, y);
    LinearModel<> model;
    LinearRegression trainer;
    trainer.train(model, data);

    std::ofstream ofs("shark-linear.dat");
    // boost::archive::polymorphic_text_oarchive oa(ofs);
    boost::archive::polymorphic_binary_oarchive oa(ofs);
    model.write(oa);
  }

  std::ifstream ifs("shark-linear.dat");
  // boost::archive::polymorphic_text_iarchive ia(ifs);
  boost::archive::polymorphic_binary_iarchive ia(ifs);
  LinearModel<> model;
  model.read(ia);

  std::cout << "Target values: \n";
  std::vector<RealVector> new_x_data;
  for (size_t i = 0; i < 5; ++i) {
    new_x_data.push_back({static_cast<double>(i)});
    std::cout << func(i) << std::endl;
  }
  auto prediction = model(createDataFromRange(new_x_data));
  std::cout << "Predictions: \n" << prediction << std::endl;

  return 0;
}
