#include <plot.h>

#define SHARK_CV_VERBOSE 1
#include <shark/Algorithms/KMeans.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Classifier.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/NearestNeighborModel.h>
#include <shark/Models/OneVersusOneClassifier.h>
#include <shark/Models/Trees/KDTree.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

#include <experimental/filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::experimental::filesystem;

using namespace shark;

const std::vector<std::string> data_names{"dataset0.csv", "dataset1.csv",
                                          "dataset2.csv", "dataset3.csv",
                                          "dataset4.csv"};

const std::vector<std::string> colors{"red", "green", "blue", "cyan", "black"};

using DataType = double;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Classes = std::unordered_map<size_t, PointCoords>;

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

void KNNClassification(const ClassificationDataset& train,
                       const ClassificationDataset& test,
                       unsigned int num_classes,
                       const std::string& name) {
  KDTree<RealVector> tree(train.inputs());
  TreeNearestNeighbors<RealVector, unsigned int> nn_alg(train, &tree);
  const unsigned int k = 5;
  NearestNeighborModel<RealVector, unsigned int> knn(&nn_alg, k);

  // compute errors
  ZeroOneLoss<unsigned int> loss;
  Data<unsigned int> predictions = knn(test.inputs());
  double accuracy = 1. - loss.eval(test.labels(), predictions);

  Classes classes;
  for (std::size_t i = 0; i != test.numberOfElements(); i++) {
    auto cluser_idx = predictions.element(i);
    auto element = test.inputs().element(i);
    classes[cluser_idx].first.push_back(element(0));
    classes[cluser_idx].second.push_back(element(1));
  }

  PlotClasses(classes, "kNN " + std::to_string(accuracy),
              name + "-knn-sharkml.png");
}

void LRClassification(const ClassificationDataset& train,
                      const ClassificationDataset& test,
                      unsigned int num_classes,
                      const std::string& name) {
  OneVersusOneClassifier<RealVector> ovo;
  unsigned int pairs = num_classes * (num_classes - 1) / 2;
  std::vector<LinearClassifier<RealVector> > lr(pairs);
  for (std::size_t n = 0, cls1 = 1; cls1 < num_classes; cls1++) {
    std::vector<OneVersusOneClassifier<RealVector>::binary_classifier_type*>
        ovo_classifiers;
    for (std::size_t cls2 = 0; cls2 < cls1; cls2++, n++) {
      // get the binary subproblem
      ClassificationDataset binary_cls_data =
          binarySubProblem(train, cls2, cls1);

      // train the binary machine
      LogisticRegression<RealVector> trainer;
      trainer.train(lr[n], binary_cls_data);
      ovo_classifiers.push_back(&lr[n]);
    }
    ovo.addClass(ovo_classifiers);
  }

  // compute errors
  ZeroOneLoss<unsigned int> loss;
  Data<unsigned int> output = ovo(test.inputs());
  double accuracy = 1. - loss.eval(test.labels(), output);

  Classes classes;
  for (std::size_t i = 0; i != test.numberOfElements(); i++) {
    auto cluser_idx = output.element(i);
    auto element = test.inputs().element(i);
    classes[cluser_idx].first.push_back(element(0));
    classes[cluser_idx].second.push_back(element(1));
  }

  PlotClasses(classes, "Logistic Regression " + std::to_string(accuracy),
              name + "-logreg-sharkml.png");
}

void SVMClassification(const ClassificationDataset& train,
                       const ClassificationDataset& test,
                       unsigned int num_classes,
                       const std::string& name) {
  double c = 10.0;
  double gamma = 0.5;
  GaussianRbfKernel<> kernel(gamma);
  OneVersusOneClassifier<RealVector> ovo;
  unsigned int pairs = num_classes * (num_classes - 1) / 2;
  std::vector<KernelClassifier<RealVector> > svm(pairs);
  for (std::size_t n = 0, cls1 = 1; cls1 < num_classes; cls1++) {
    std::vector<OneVersusOneClassifier<RealVector>::binary_classifier_type*>
        ovo_classifiers;
    for (std::size_t cls2 = 0; cls2 < cls1; cls2++, n++) {
      // get the binary subproblem
      ClassificationDataset binary_cls_data =
          binarySubProblem(train, cls2, cls1);

      // train the binary machine
      CSvmTrainer<RealVector> trainer(&kernel, c, false);
      trainer.train(svm[n], binary_cls_data);
      ovo_classifiers.push_back(&svm[n]);
    }
    ovo.addClass(ovo_classifiers);
  }

  // compute errors
  ZeroOneLoss<unsigned int> loss;
  Data<unsigned int> output = ovo(test.inputs());
  double accuracy = 1. - loss.eval(test.labels(), output);

  Classes classes;
  for (std::size_t i = 0; i != test.numberOfElements(); i++) {
    auto cluser_idx = output.element(i);
    auto element = test.inputs().element(i);
    classes[cluser_idx].first.push_back(element(0));
    classes[cluser_idx].second.push_back(element(1));
  }

  PlotClasses(classes, "SVM " + std::to_string(accuracy),
              name + "-svm-sharkml.png");
}

int main(int argc, char** argv) {
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);
    for (auto& dataset : data_names) {
      auto dataset_name = base_dir / dataset;
      if (fs::exists(dataset_name)) {
        ClassificationDataset data;
        importCSV(data, dataset_name, LabelPosition::LAST_COLUMN);
        data = selectInputFeatures(
            data, std::vector<int>{1, 2});  // exclude index column

        std::size_t num_samples = data.numberOfElements();
        std::size_t num_features = dataDimension(data.inputs());
        std::size_t num_classes = numberOfClasses(data.labels());

        std::cout << dataset << "\n"
                  << "Num samples: " << num_samples
                  << " num features: " << num_features
                  << " num clusters: " << num_classes << std::endl;

        // split data set in the training and testing parts
        ClassificationDataset test_data = splitAtElement(data, 1200);

        // create data set for multiclass problem
        repartitionByClass(data);

        SVMClassification(data, test_data, num_classes, dataset);
        LRClassification(data, test_data, num_classes, dataset);
        KNNClassification(data, test_data, num_classes, dataset);
      } else {
        std::cerr << "Dataset file " << dataset_name << " missed\n";
      }
    }
  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  return 0;
}
