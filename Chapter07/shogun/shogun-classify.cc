#include <plot.h>

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/io/File.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/normalizer/ZeroMeanCenterKernelNormalizer.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/multiclass/MulticlassLogisticRegression.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/util/factory.h>

#include <experimental/filesystem>
#include <iostream>
#include <map>

namespace fs = std::experimental::filesystem;

using namespace shogun;
using DataType = float64_t;
using Matrix = SGMatrix<DataType>;

const std::vector<std::string> data_names{"dataset0.csv", "dataset1.csv",
                                          "dataset2.csv", "dataset3.csv",
                                          "dataset4.csv"};

const std::vector<std::string> colors{"red", "green", "blue", "cyan", "black"};

using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Classes = std::map<index_t, PointCoords>;

void PlotClasses(const Classes& classes,
                 const std::string& name,
                 const std::string& file_name) {
  plotcpp::Plot plt(true);
  // plt.SetTerminal("qt");
  plt.SetTerminal("png");
  plt.SetOutput(file_name);
  plt.SetTitle(name);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
  for (auto& cls : classes) {
    std::stringstream params;
    params << "lc rgb '" << colors[cls.first] << "' pt 7";
    plt.AddDrawing(
        draw_state,
        plotcpp::Points(cls.second.first.begin(), cls.second.first.end(),
                        cls.second.second.begin(),
                        std::to_string(cls.first) + " cls", params.str()));
  }

  plt.EndDraw2D(draw_state);
  plt.Flush();
}

void DrawDataSet(Some<CDenseFeatures<DataType>> features,
                 Some<CMulticlassLabels> labels,
                 const std::string& title,
                 const std::string& name) {
  Classes classes;
  auto feature_matrix = features->get_feature_matrix();
  for (index_t i = 0; i < labels->get_num_labels(); ++i) {
    auto label_idx_pred = labels->get_label(i);
    auto vector = feature_matrix.get_column(i);
    classes[label_idx_pred].first.push_back(vector[0]);
    classes[label_idx_pred].second.push_back(vector[1]);
  }

  PlotClasses(classes, title, name);
}

void KNNClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CMulticlassLabels> labels,
                       Some<CDenseFeatures<DataType>> test_features,
                       Some<CMulticlassLabels> test_labels,
                       const std::string& name) {
  int32_t k = 3;
  auto distance = some<CEuclideanDistance>(features, features);
  auto knn = some<CKNN>(k, distance, labels);
  knn->train();

  // evaluate model on test data
  auto new_labels = wrap(knn->apply_multiclass(test_features));

  auto eval_criterium = some<CMulticlassAccuracy>();
  auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
  std::cout << "knn " << name << " accuracy = " << accuracy << std::endl;

  DrawDataSet(test_features, new_labels, "kNN " + std::to_string(accuracy),
              name + "-knn-shogun.png");
}

void LogClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CMulticlassLabels> labels,
                       Some<CDenseFeatures<DataType>> test_features,
                       Some<CMulticlassLabels> test_labels,
                       const std::string& name) {
  auto log_reg = some<CMulticlassLogisticRegression>();

  // search for hyper-parameters
  auto root = some<CModelSelectionParameters>();
  // z - regularization
  CModelSelectionParameters* z = new CModelSelectionParameters("m_z");
  root->append_child(z);
  z->build_values(0.2, 1.0, R_LINEAR, 0.1);

  index_t k = 3;
  CStratifiedCrossValidationSplitting* splitting =
      new CStratifiedCrossValidationSplitting(labels, k);

  auto eval_criterium = some<CMulticlassAccuracy>();

  auto cross = some<CCrossValidation>(log_reg, features, labels, splitting,
                                      eval_criterium);
  cross->set_num_runs(1);

  auto model_selection = some<CGridSearchModelSelection>(cross, root);
  CParameterCombination* best_params =
      wrap(model_selection->select_model(false));
  best_params->apply_to_machine(log_reg);
  best_params->print_tree();

  // train
  log_reg->set_labels(labels);
  log_reg->train(features);

  // evaluate model on test data
  auto new_labels = wrap(log_reg->apply_multiclass(test_features));

  auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
  std::cout << "logistic regression " << name << " accuracy = " << accuracy
            << std::endl;

  DrawDataSet(test_features, new_labels,
              "Logistic regression " + std::to_string(accuracy),
              name + "-logreg-shogun.png");
}

void SVMClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CMulticlassLabels> labels,
                       Some<CDenseFeatures<DataType>> test_features,
                       Some<CMulticlassLabels> test_labels,
                       const std::string& name) {
  auto kernel = some<CGaussianKernel>(features, features, 5);
  // auto kernel = some<CLinearKernel>(features, features);
  // one vs one classification
  auto svm = some<CMulticlassLibSVM>(LIBSVM_C_SVC);
  svm->set_kernel(kernel);

  // search for hyper-parameters
  auto root = some<CModelSelectionParameters>();
  // C - how much you want to avoid misclassifying
  CModelSelectionParameters* c = new CModelSelectionParameters("C");
  root->append_child(c);
  c->build_values(1.0, 1000.0, R_LINEAR, 100.);

  auto params_kernel = some<CModelSelectionParameters>("kernel", kernel);
  root->append_child(params_kernel);

  auto params_kernel_width =
      some<CModelSelectionParameters>("combined_kernel_weight");
  params_kernel_width->build_values(0.1, 10.0, R_LINEAR, 0.5);

  params_kernel->append_child(params_kernel_width);

  index_t k = 3;
  CStratifiedCrossValidationSplitting* splitting =
      new CStratifiedCrossValidationSplitting(labels, k);

  auto eval_criterium = some<CMulticlassAccuracy>();

  auto cross =
      some<CCrossValidation>(svm, features, labels, splitting, eval_criterium);
  cross->set_num_runs(1);

  auto model_selection = some<CGridSearchModelSelection>(cross, root);
  CParameterCombination* best_params =
      wrap(model_selection->select_model(false));
  best_params->apply_to_machine(svm);
  best_params->print_tree();

  // train SVM
  svm->set_labels(labels);
  svm->train(features);

  // evaluate model on test data
  auto new_labels = wrap(svm->apply_multiclass(test_features));

  auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
  std::cout << "svm " << name << " accuracy = " << accuracy << std::endl;

  DrawDataSet(test_features, new_labels, "SVM " + std::to_string(accuracy),
              name + "-svm-shogun.png");
}

int main(int argc, char** argv) {
  init_shogun_with_defaults();
  // shogun::sg_io->set_loglevel(shogun::MSG_INFO);
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);
    for (auto& dataset : data_names) {
      auto dataset_name = base_dir / dataset;
      if (fs::exists(dataset_name)) {
        auto csv_file = some<CCSVFile>(dataset_name.string().c_str());
        Matrix data;
        data.load(csv_file);

        // Exclude cluster and index info from data
        // Shogun csv loader loads matrixes in column major order
        Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);
        Matrix inputs = data.submatrix(1, data.num_cols - 1);  // make a view
        inputs = inputs.clone();                // copy exact data
        Matrix outputs = data.submatrix(3, 4);  // make a view
        outputs = outputs.clone();              // copy exact data

        // Transpose back because shogun algorithms expect that samples are in
        // columns
        Matrix::transpose_matrix(inputs.matrix, inputs.num_rows,
                                 inputs.num_cols);

        Matrix::transpose_matrix(outputs.matrix, outputs.num_rows,
                                 outputs.num_cols);

        // split data to train and test data sets
        int test_n = 300;
        Matrix test_inputs = inputs.submatrix(0, test_n).clone();
        inputs = inputs.submatrix(test_n, inputs.num_cols).clone();

        Matrix test_outputs = outputs.submatrix(0, test_n).clone();
        outputs = outputs.submatrix(test_n, outputs.num_cols).clone();

        // create a datasets
        auto features = some<CDenseFeatures<DataType>>(inputs);
        auto labels = some<CMulticlassLabels>(outputs.get_row_vector(0));

        auto test_features = some<CDenseFeatures<DataType>>(test_inputs);
        auto test_labels =
            some<CMulticlassLabels>(test_outputs.get_row_vector(0));

        // rescale features
        auto scaler = shogun::wrap(new shogun::CRescaleFeatures());
        scaler->fit(features);
        scaler->transform(features);
        scaler->transform(test_features);

        // print statistics
        auto num_classes = labels->get_num_classes();
        std::cout << "Dataset : " << dataset << "\n";
        std::cout << "Num features per sample : "
                  << features->get_num_features() << "\n";
        std::cout << "Num samples : " << features->get_num_vectors()
                  << std::endl;
        std::cout << "Num classes : " << num_classes << std::endl;

        SVMClassification(features, labels, test_features, test_labels,
                          dataset);
        LogClassification(features, labels, test_features, test_labels,
                          dataset);

        KNNClassification(features, labels, test_features, test_labels,
                          dataset);

        DrawDataSet(features, labels, dataset, dataset + ".png");
      } else {
        std::cerr << "Dataset file " << dataset_name << " missed\n";
      }
    }
  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  exit_shogun();
  return 0;
}
