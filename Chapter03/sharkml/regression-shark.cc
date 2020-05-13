#include "data.h"
#include "plot.h"
#include "polynomial-model.h"
#include "polynomial-regression.h"

#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Data/CVDatasetTools.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Normalizer.h>

using namespace shark;

int main() {
  Data<RealVector> x;
  Data<RealVector> y;
  std::tie(x, y) = GetXYData();
  RegressionDataset train_data(x, y);
  train_data.shuffle();

  Data<RealVector> val_x;
  Data<RealVector> val_y;
  std::tie(val_x, val_y) = GetXYValidationData();
  RegressionDataset validation_data(val_x, val_y);

  auto x_minmax = DataMin(train_data.inputs());

  // normalization
  bool remove_mean = true;
  shark::Normalizer<shark::RealVector> x_normalizer;
  shark::NormalizeComponentsUnitVariance<shark::RealVector> normalizing_trainer(
      remove_mean);
  normalizing_trainer.train(x_normalizer, train_data.inputs());
  train_data = transformInputs(train_data, x_normalizer);
  validation_data = transformInputs(validation_data, x_normalizer);

  shark::Normalizer<shark::RealVector> y_normalizer;
  normalizing_trainer.train(y_normalizer, train_data.labels());
  train_data = transformLabels(train_data, x_normalizer);
  validation_data = transformLabels(validation_data, x_normalizer);

  // Train the model

  // try to change this factor to be greater than zero for polynomial dergees
  // higher 10 to see how regularization beat high variance
  double regularization_factor = 0.0;

  // use degree of 1 - 4 to see the high bias effect
  // use degree of 10 - 15 to see the high variance effect
  double polynomial_degree = 8;

  int num_epochs = 300;

  PolynomialModel<> model;
  PolynomialRegression trainer(regularization_factor, polynomial_degree,
                               num_epochs);
  PolynomialRegression::MonitorType monitor;
  monitor.validation_data = validation_data;
  trainer.setMonitor(&monitor);
  trainer.train(model, train_data);

  // Show metrics
  SquaredLoss<> mse_loss;
  Data<RealVector> prediction = model(train_data.inputs());
  auto mse = mse_loss(train_data.labels(), prediction);
  auto rmse = std::sqrt(mse);
  AbsoluteLoss<> abs_loss;
  auto mae = abs_loss(train_data.labels(), prediction);
  // R^2
  auto var = shark::variance(train_data.labels());
  auto r_squared = 1 - mse / var(0);
  // Adjusted R^2
  auto n = train_data.labels().numberOfElements();
  auto k = std::pow(train_data.inputs().shape()[0] + 1, n);
  auto adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1));

  std::cout << "Mean Squared Error :\n" << mse << std::endl;
  std::cout << "Root Mean Squared Error :\n" << rmse << std::endl;
  std::cout << "Root Absolute Error :\n" << mae << std::endl;
  std::cout << "R Squared :\n" << r_squared << std::endl;
  std::cout << "Adjusted R Squared :\n" << adj_r_squared << std::endl;

  // plot the data
  auto new_x = LinSpace(x_minmax.first, x_minmax.second, 50);
  new_x = x_normalizer(new_x);
  prediction = model(new_x);

  PlotData(train_data.inputs(), train_data.labels(), validation_data.inputs(),
           validation_data.labels(), new_x, prediction);
  PlotTrain(monitor.train_steps, monitor.validation_steps);

  return 0;
}
