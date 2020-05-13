#include <plot.h>

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/clustering/GMM.h>
#include <shogun/clustering/Hierarchical.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/io/File.h>
#include <shogun/labels/MulticlassLabels.h>
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

const std::vector<std::string> data_names{"dataset0.csv", "dataset1.csv",
                                          "dataset2.csv", "dataset3.csv",
                                          "dataset4.csv", "dataset5.csv"};

const std::vector<std::string> colors{"black", "red", "blue", "green", "cyan"};

using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<index_t, PointCoords>;

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

void MakeGMMClustering(Some<CDenseFeatures<DataType>> features,
                       const int num_clusters,
                       const std::string& name) {
  std::cout << "GMM\n";
  auto gmm = some<CGMM>(num_clusters);
  gmm->set_features(features);
  gmm->train_em();

  Clusters clusters;
  auto feature_matrix = features->get_feature_matrix();
  for (index_t i = 0; i < features->get_num_vectors(); ++i) {
    auto vector = feature_matrix.get_column(i);
    auto log_likelihoods = gmm->cluster(vector);
    auto max_el = std::max_element(log_likelihoods.begin(),
                                   std::prev(log_likelihoods.end()));
    auto label_idx =
        static_cast<int>(std::distance(log_likelihoods.begin(), max_el));
    clusters[label_idx].first.push_back(vector[0]);
    clusters[label_idx].second.push_back(vector[1]);
  }

  PlotClusters(clusters, "GMM", name + "-gmm.png");
}

void MakeKMeansClustering(Some<CDenseFeatures<DataType>> features,
                          const int num_clusters,
                          const std::string& name) {
  std::cout << "K-Means\n";
  auto distance = some<CEuclideanDistance>(features, features);
  auto clustering = some<CKMeans>(num_clusters, distance.get());
  clustering->train(features);
  std::cout << "Cluster centers :\n";
  clustering->get_cluster_centers().display_matrix();

  Clusters clusters;
  auto feature_matrix = features->get_feature_matrix();
  CMulticlassLabels* result = clustering->apply()->as<CMulticlassLabels>();
  for (index_t i = 0; i < result->get_num_labels(); ++i) {
    auto label_idx = static_cast<int>(result->get_label(i));
    auto vector = feature_matrix.get_column(i);
    clusters[label_idx].first.push_back(vector[0]);
    clusters[label_idx].second.push_back(vector[1]);
  }

  PlotClusters(clusters, "K-Means", name + "-kmeans.png");
}

int main(int argc, char** argv) {
  init_shogun_with_defaults();
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
        // colums
        Matrix::transpose_matrix(inputs.matrix, inputs.num_rows,
                                 inputs.num_cols);

        // create a dataset
        auto features = some<CDenseFeatures<DataType>>(inputs);
        auto cluster_labels = some<CMulticlassLabels>(outputs.get_column(0));
        auto num_clusters = cluster_labels->get_num_classes();
        if (num_clusters > 3 || num_clusters < 2)
          num_clusters = 3;

        // print statistics
        std::cout << "Dataset : " << dataset << "\n";
        std::cout << "Num features per sample : "
                  << features->get_num_features() << "\n";
        std::cout << "Num samples : " << features->get_num_vectors()
                  << std::endl;
        std::cout << "Num clusters : " << num_clusters << std::endl;

        MakeKMeansClustering(features, num_clusters, dataset);
        MakeGMMClustering(features, num_clusters, dataset);
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
