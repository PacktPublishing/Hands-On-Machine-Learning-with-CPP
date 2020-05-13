#include "data/data.h"
#include <plot.h>

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/util/factory.h>

#include <experimental/filesystem>
#include <iostream>
#include <map>

namespace fs = std::experimental::filesystem;

using namespace shogun;
using Matrix = shogun::SGMatrix<DataType>;
using Vector = shogun::SGVector<DataType>;

std::pair<Matrix, Vector> GenerateShogunData(double s,
                                             double e,
                                             size_t n,
                                             size_t seed,
                                             bool noise) {
  Values x, y;
  std::tie(x, y) = GenerateData(s, e, n, seed, noise);
  Matrix x_values(1, static_cast<int>(n));
  Vector y_values(static_cast<int>(n));

  for (size_t i = 0; i < n; ++i) {
    x_values.set_element(x[i], 0, static_cast<int>(i));
    y_values.set_element(y[i], static_cast<int>(i));
  }
  return {x_values, y_values};
}

void PlotResults(Some<CDenseFeatures<DataType>> test_features,
                 Some<CRegressionLabels> test_labels,
                 Some<CRegressionLabels> pred_labels,
                 const std::string& title,
                 const std::string& file_name) {
  auto x_coords = test_features->get_feature_matrix();
  auto y_coords = test_labels->get_labels();
  auto y_pred_coords = pred_labels->get_labels();

  plotcpp::Plot plt;
  plt.SetTerminal("png");
  plt.SetOutput(file_name + ".png");
  plt.SetTitle(title);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  plt.Draw2D(
      plotcpp::Points(x_coords.begin(), x_coords.end(), y_coords.begin(),
                      "orig", "lc rgb 'black' pt 7"),
      plotcpp::Lines(x_coords.begin(), x_coords.end(), y_pred_coords.begin(),
                     "pred", "lc rgb 'red' lw 2"));
  plt.Flush();
}

void GBMClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CRegressionLabels> labels,
                       Some<CDenseFeatures<DataType>> test_features,
                       Some<CRegressionLabels> test_labels) {
  // mark feature type as continuous
  SGVector<bool> feature_type(1);
  feature_type.set_const(false);
  /*
   * A CART tree is a binary decision tree that is constructed by splitting a
   * node into two child nodes repeatedly, beginning with the root node that
   * contains the whole dataset.
   */
  auto tree = some<CCARTree>(feature_type, PT_REGRESSION);
  // try to change tree depth to see its influence on accuracy
  tree->set_max_depth(3);
  auto loss = some<CSquaredLoss>();

  // GBM supports only regression
  // try to change learning rate to see its influence on accuracy
  auto sgbm = some<CStochasticGBMachine>(tree, loss, /*iterations*/ 100,
                                         /*learning rate*/ 0.1, 1.0);
  sgbm->set_labels(labels);
  sgbm->train(features);

  // evaluate model on test data
  auto new_labels = wrap(sgbm->apply_regression(test_features));

  auto eval_criterium = some<CMeanSquaredError>();
  auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
  std::cout << "GBM classification accuracy = " << accuracy << std::endl;

  PlotResults(test_features, test_labels, new_labels,
              "Shogun Gradient Boosting", "shogun-gbm");
}

void RFClassification(Some<CDenseFeatures<DataType>> features,
                      Some<CRegressionLabels> labels,
                      Some<CDenseFeatures<DataType>> test_features,
                      Some<CRegressionLabels> test_labels) {
  // number of attributes chosen randomly during node split in candidate trees
  int32_t num_rand_feats = 1;
  // number of trees in forest
  int32_t num_bags = 10;

  auto rand_forest =
      shogun::some<shogun::CRandomForest>(num_rand_feats, num_bags);

  auto vote = shogun::some<shogun::CMajorityVote>();
  rand_forest->set_combination_rule(vote);
  // mark feature type as continuous
  SGVector<bool> feature_type(1);
  feature_type.set_const(false);
  rand_forest->set_feature_types(feature_type);

  rand_forest->set_labels(labels);
  rand_forest->set_machine_problem_type(PT_REGRESSION);
  rand_forest->train(features);

  // evaluate model on test data
  auto new_labels = wrap(rand_forest->apply_regression(test_features));

  auto eval_criterium = some<CMeanSquaredError>();
  auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
  std::cout << "RF classification accuracy = " << accuracy << std::endl;

  PlotResults(test_features, test_labels, new_labels, "Shogun Random Forest",
              "shogun-rf");
}

int main(int /*argc*/, char** /*argv*/) {
  init_shogun_with_defaults();
  // shogun::sg_io->set_loglevel(shogun::MSG_INFO);

  // generate data
  const size_t seed = 3463;
  const size_t num_samples = 1000;
  SGMatrix<DataType> x_values;
  SGVector<DataType> y_values;
  std::tie(x_values, y_values) =
      GenerateShogunData(-10, 10, num_samples, seed, true);
  auto train_features = some<CDenseFeatures<DataType>>(x_values);
  auto train_labels = some<CRegressionLabels>(y_values);

  std::tie(x_values, y_values) =
      GenerateShogunData(-10, 10, num_samples, seed, false);
  auto test_features = some<CDenseFeatures<DataType>>(x_values);
  auto test_labels = some<CRegressionLabels>(y_values);

  GBMClassification(train_features, train_labels, test_features, test_labels);
  RFClassification(train_features, train_labels, test_features, test_labels);

  exit_shogun();
  return 0;
}
