#include <plot.h>

#define SHARK_CV_VERBOSE 1
#include <shark/Algorithms/Trainers/OneClassSvmTrainer.h>
#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

#include <experimental/filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::experimental::filesystem;

using namespace shark;

const std::vector<std::string> colors{"black", "red", "blue", "green", "cyan"};

using DataType = double;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<size_t, PointCoords>;

void PlotClusters(const Clusters& clusters,
                  const std::string& name,
                  const std::string& file_name) {
  plotcpp::Plot plt(true);
  // plt.SetTerminal("qt");
  plt.SetTerminal("png");
  plt.SetOutput(file_name);
  plt.SetTitle(name);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  // plt.SetAutoscale();
  plt.GnuplotCommand("set size square");
  plt.GnuplotCommand("set grid");

  auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
  for (auto& cluster : clusters) {
    std::stringstream params;
    params << "lc rgb '" << colors[cluster.first] << "' pt 7";
    plt.AddDrawing(draw_state,
                   plotcpp::Points(
                       cluster.second.first.begin(), cluster.second.first.end(),
                       cluster.second.second.begin(),
                       std::to_string(cluster.first) + " cls", params.str()));
  }

  plt.EndDraw2D(draw_state);
  plt.Flush();
}

int main(int argc, char** argv) {
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);
    std::string data_name_multi{"multivar.csv"};
    auto dataset_name = base_dir / data_name_multi;
    if (fs::exists(dataset_name)) {
      UnlabeledData<RealVector> data;
      importCSV(data, dataset_name);

      data.splitBatch(0, 50);
      auto test_data = data.splice(1);

      std::size_t num_samples = data.numberOfElements();
      std::size_t num_features = dataDimension(data.inputs());

      std::cout << "Num samples: " << num_samples
                << " num features: " << num_features << std::endl;

      double gamma = 0.5;  // kernel bandwidth parameter
      GaussianRbfKernel<> kernel(gamma);
      KernelExpansion<RealVector> ke(&kernel);

      double nu = 0.5;  // parameter of the method for controlling the
                        // smoothness of the solution

      OneClassSvmTrainer<RealVector> trainer(&kernel, nu);
      trainer.stoppingCondition().minAccuracy = 1e-6;
      trainer.train(ke, data);

      double dist_threshold = -0.2;
      Clusters plot_clusters;
      RealVector output;
      auto detect = [&](const UnlabeledData<RealVector>& data) {
        for (size_t i = 0; i < data.numberOfElements(); ++i) {
          ke.eval(data.element(i), output);
          auto x = data.element(i)[0];
          auto y = data.element(i)[1];
          if (output[0] > dist_threshold) {
            plot_clusters[0].first.push_back(x);
            plot_clusters[0].second.push_back(y);
          } else {
            plot_clusters[1].first.push_back(x);
            plot_clusters[1].second.push_back(y);
          }
        }
      };
      detect(data);
      detect(test_data);

      PlotClusters(plot_clusters, "One Class Svm", "shark-ocsvm.png");

    } else {
      std::cerr << "Dataset file " << dataset_name << " missed\n";
    }
  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  return 0;
}
