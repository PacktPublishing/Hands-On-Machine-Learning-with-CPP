#include <plot.h>

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/classifier/svm/LibSVMOneClass.h>
#include <shogun/io/File.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/util/factory.h>

#include <experimental/filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::experimental::filesystem;

using namespace shogun;
using DataType = float64_t;
using Matrix = SGMatrix<DataType>;

const std::vector<std::string> colors{"black", "red", "blue", "green", "cyan"};

using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<index_t, PointCoords>;

void PlotClusters(const Clusters& clusters,
                  const std::string& name,
                  const std::string& file_name) {
  plotcpp::Plot plt(true);
  plt.SetTerminal("png");
  // plt.SetTerminal("qt");
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
  init_shogun_with_defaults();
  if (argc > 1) {
    auto base_dir = fs::path(argv[1]);
    std::string data_name_multi{"multivar.csv"};
    auto dataset_name = base_dir / data_name_multi;
    if (fs::exists(dataset_name)) {
      auto csv_file = some<CCSVFile>(dataset_name.string().c_str());
      Matrix data;
      data.load(csv_file);

      Matrix train = data.submatrix(0, 50);
      train = train.clone();
      Matrix test = data.submatrix(50, data.num_cols);
      test = test.clone();

      // create a dataset
      auto features = some<CDenseFeatures<DataType>>(train);
      auto test_features = some<CDenseFeatures<DataType>>(test);

      auto gauss_kernel = some<CGaussianKernel>(features, features, 0.5);

      auto c = 0.5;
      auto svm = some<CLibSVMOneClass>(c, gauss_kernel);
      svm->train(features);

      double dist_threshold = -3.15;
      Clusters plot_clusters;

      auto detect = [&](Some<CDenseFeatures<DataType>> data) {
        auto labels = svm->apply(data);
        for (int i = 0; i < labels->get_num_labels(); ++i) {
          auto dist = labels->get_value(i);
          auto vec = data->get_feature_vector(i);
          if (dist > dist_threshold) {
            plot_clusters[0].first.push_back(vec[0]);
            plot_clusters[0].second.push_back(vec[1]);
          } else {
            plot_clusters[1].first.push_back(vec[0]);
            plot_clusters[1].second.push_back(vec[1]);
          }
        }
      };

      detect(features);
      detect(test_features);

      PlotClusters(plot_clusters, "One Class Svm", "shogun-ocsvm.png");

    } else {
      std::cerr << "Dataset file " << dataset_name << " missed\n";
    }

  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  exit_shogun();
  return 0;
}
