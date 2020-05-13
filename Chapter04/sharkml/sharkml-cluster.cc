#include <plot.h>

#define SHARK_CV_VERBOSE 1
#include <shark/Algorithms/KMeans.h>
#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Clustering/HardClusteringModel.h>
#include <shark/Models/Clustering/HierarchicalClustering.h>
#include <shark/Models/Trees/LCTree.h>

#include <experimental/filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::experimental::filesystem;

using namespace shark;

const std::vector<std::string> data_names{"dataset0.csv", "dataset1.csv",
                                          "dataset2.csv", "dataset3.csv",
                                          "dataset4.csv", "dataset5.csv"};

const std::vector<std::string> colors{"black", "red", "blue", "green", "cyan"};

using DataType = double;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<size_t, PointCoords>;

void PlotClusters(const Clusters& clusters,
                  const std::string& name,
                  const std::string& file_name) {
  plotcpp::Plot plt;
  plt.SetTerminal("png");
  plt.SetOutput(file_name);
  plt.SetTitle(name);
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
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

void MakeHierarhicalClustering(UnlabeledData<RealVector>& features,
                               const int num_clusters,
                               const std::string& name) {
  LCTree<RealVector> tree(
      features,
      TreeConstruction(0, features.numberOfElements() / num_clusters));
  HierarchicalClustering<RealVector> clustering(&tree);
  HardClusteringModel<RealVector> model(&clustering);

  std::cout << "num nodes: " << tree.nodes() << std::endl;
  std::cout << "num clusters: " << clustering.numberOfClusters() << std::endl;
  Data<unsigned> clusters = model(features);

  Clusters plot_clusters;
  for (std::size_t i = 0; i != features.numberOfElements(); i++) {
    auto cluser_idx = clusters.element(i);
    auto element = features.element(i);
    plot_clusters[cluser_idx].first.push_back(element(0));
    plot_clusters[cluser_idx].second.push_back(element(1));
  }

  PlotClusters(plot_clusters, "Hierarchical", name + "-hierarchical.png");
}

void MakeKMeansClustering(UnlabeledData<RealVector>& features,
                          const int num_clusters,
                          const std::string& name) {
  Centroids centroids;
  kMeans(features, num_clusters, centroids);

  HardClusteringModel<RealVector> model(&centroids);
  Data<unsigned> clusters = model(features);

  Clusters plot_clusters;
  for (std::size_t i = 0; i != features.numberOfElements(); i++) {
    auto cluser_idx = clusters.element(i);
    auto element = features.element(i);
    plot_clusters[cluser_idx].first.push_back(element(0));
    plot_clusters[cluser_idx].second.push_back(element(1));
  }

  PlotClusters(plot_clusters, "K-Means", name + "-kmeans.png");
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
        std::size_t num_clusters = numberOfClasses(data.labels());
        if (num_clusters < 2)
          num_clusters = 3;

        std::cout << dataset << "\n"
                  << "Num samples: " << num_samples
                  << " num features: " << num_features
                  << " num clusters: " << num_clusters << std::endl;

        MakeKMeansClustering(data.inputs(), num_clusters, dataset);
        MakeHierarhicalClustering(data.inputs(), num_clusters, dataset);
      } else {
        std::cerr << "Dataset file " << dataset_name << " missed\n";
      }
    }
  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  return 0;
}
