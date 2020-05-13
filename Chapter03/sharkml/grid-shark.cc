#include "data.h"
#include "plot.h"
#include "polynomial-model.h"
#include "polynomial-regression.h"

#define SHARK_CV_VERBOSE 1
#include <shark/Algorithms/DirectSearch/GridSearch.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Data/CVDatasetTools.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Normalizer.h>
#include <shark/ObjectiveFunctions/CrossValidationError.h>

using namespace shark;

int main() {
  Data<RealVector> x;
  Data<RealVector> y;
  std::tie(x, y) =
      GenerateData(/*num_samples*/ 500, /*no_noise*/ false, /*seed*/ 7568);
  RegressionDataset train_data(x, y);
  train_data.shuffle();

  auto x_minmax = DataMin(train_data.inputs());

  // normalization
  bool remove_mean = true;
  shark::Normalizer<shark::RealVector> x_normalizer;
  shark::NormalizeComponentsUnitVariance<shark::RealVector> normalizing_trainer(
      remove_mean);
  normalizing_trainer.train(x_normalizer, train_data.inputs());
  train_data = transformInputs(train_data, x_normalizer);

  // split dataset
  const unsigned int num_folds = 5;
  CVFolds<RegressionDataset> folds =
      createCVSameSize<RealVector, RealVector>(train_data, num_folds);

  // Grid search the model
  AbsoluteLoss<> loss;
  // SquaredLoss<> loss;
  double regularization_factor = 0.0;
  double polynomial_degree = 8;
  int num_epochs = 300;
  PolynomialModel<> model;
  PolynomialRegression trainer(regularization_factor, polynomial_degree,
                               num_epochs);

  CrossValidationError<PolynomialModel<>, RealVector> cv_error(
      folds, &trainer, &model, &trainer, &loss);

  GridSearch grid;
  std::vector<double> min(2);
  std::vector<double> max(2);
  std::vector<size_t> sections(2);
  // regularization factor
  min[0] = 0.0;
  max[0] = 0.00001;
  sections[0] = 6;
  // polynomial degree
  min[1] = 4;
  max[1] = 10.0;
  sections[1] = 6;
  grid.configure(min, max, sections);
  grid.step(cv_error);

  // train final model
  std::cout << grid.solution() << std::endl;
  trainer.setParameterVector(grid.solution().point);
  trainer.train(model, train_data);

  // plot the data
  auto new_x = LinSpace(x_minmax.first, x_minmax.second, 50);
  new_x = x_normalizer(new_x);
  Data<RealVector> prediction = model(new_x);

  PlotData(train_data.inputs(), train_data.labels(),
           folds.validation(0).inputs(), folds.validation(0).labels(), new_x,
           prediction);

  return 0;
}
