// http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html

#include <plot.h>

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/converter/FactorAnalysis.h>
#include <shogun/converter/Isomap.h>
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/converter/TDistributedStochasticNeighborEmbedding.h>
#include <shogun/converter/ica/FastICA.h>
#include <shogun/io/File.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/preprocessor/KernelPCA.h>
#include <shogun/preprocessor/PCA.h>
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
using PointCoords3d = std::tuple<Coords, Coords, Coords>;
using Clusters = std::unordered_map<index_t, PointCoords>;
using Clusters3d = std::unordered_map<index_t, PointCoords3d>;

const std::string data_file_name{"swissroll.dat"};
const std::string labels_file_name{"swissroll_labels.dat"};

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

void Plot3DData(Some<CDenseFeatures<DataType>> features,
                Some<CMulticlassLabels> lables) {
  Clusters3d clusters;

  auto matrix = features->get_feature_matrix();
  for (index_t i = 0; i < features->get_num_vectors(); ++i) {
    auto vector = matrix.get_column_vector(i);
    auto label = static_cast<int>(lables->get_label(i));
    std::get<0>(clusters[label]).push_back(vector[0]);
    std::get<1>(clusters[label]).push_back(vector[1]);
    std::get<2>(clusters[label]).push_back(vector[2]);
  }

  plotcpp::Plot plt(true);
  plt.SetTerminal("png");
  // plt.SetTerminal("qt");
  plt.SetOutput("3d_data.png");
  plt.SetTitle("Swissroll data");
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  // plt.SetAutoscale();
  plt.GnuplotCommand("set size square");
  plt.GnuplotCommand("set grid");

  auto draw_state = plt.StartDraw3D<Coords::const_iterator>();
  for (auto& cluster : clusters) {
    std::stringstream params;
    params << "lc rgb '" << colors[cluster.first] << "' pt 7";
    plt.AddDrawing(draw_state,
                   plotcpp::Points3D(std::get<0>(cluster.second).begin(),
                                     std::get<0>(cluster.second).end(),
                                     std::get<1>(cluster.second).begin(),
                                     std::get<2>(cluster.second).begin(),
                                     std::to_string(cluster.first) + " cls",
                                     params.str()));
  }

  plt.EndDraw3D(draw_state);
  plt.Flush();
}

void ICAReduction(Some<CDenseFeatures<DataType>> features,
                  Some<CMulticlassLabels> lables,
                  const int target_dim) {
  auto ica = some<CFastICA>();
  ica->fit(features);

  auto new_features =
      static_cast<CDenseFeatures<DataType>*>(ica->transform(features));
  auto casted = CDenseFeatures<float64_t>::obtain_from_generic(new_features);

  Clusters clusters;
  auto unmixed_signal = casted->get_feature_matrix();
  for (index_t i = 0; i < new_features->get_num_vectors(); ++i) {
    auto new_vector = unmixed_signal.get_column(i);
    auto label = static_cast<int>(lables->get_label(i));
    // choose 1 and 2 as our main components
    clusters[label].first.push_back(new_vector[1]);
    clusters[label].second.push_back(new_vector[2]);
  }

  PlotClusters(clusters, "ICA", "ica-shogun.png");
}

void FAReduction(Some<CDenseFeatures<DataType>> features,
                 Some<CMulticlassLabels> lables,
                 const int target_dim) {
  auto fa = some<CFactorAnalysis>();
  fa->set_target_dim(target_dim);
  fa->fit(features);

  auto new_features =
      static_cast<CDenseFeatures<DataType>*>(fa->transform(features));

  Clusters clusters;
  auto feature_matrix = new_features->get_feature_matrix();
  for (index_t i = 0; i < new_features->get_num_vectors(); ++i) {
    auto new_vector = feature_matrix.get_column(i);
    auto label = static_cast<int>(lables->get_label(i));
    clusters[label].first.push_back(new_vector[1]);
    clusters[label].second.push_back(new_vector[2]);
  }

  PlotClusters(clusters, "Factor analysis", "fa-shogun.png");
}

void TSNEReduction(Some<CDenseFeatures<DataType>> features,
                   Some<CMulticlassLabels> lables,
                   const int target_dim) {
  auto tsne = some<CTDistributedStochasticNeighborEmbedding>();
  tsne->set_target_dim(target_dim);
  tsne->fit(features);

  auto new_features =
      static_cast<CDenseFeatures<DataType>*>(tsne->transform(features));

  Clusters clusters;
  auto feature_matrix = new_features->get_feature_matrix();
  for (index_t i = 0; i < new_features->get_num_vectors(); ++i) {
    auto new_vector = feature_matrix.get_column(i);
    auto label = static_cast<int>(lables->get_label(i));
    clusters[label].first.push_back(new_vector[1]);
    clusters[label].second.push_back(new_vector[2]);
  }

  PlotClusters(clusters, "t-SNE", "tsne-shogun.png");
}

void MDSReduction(Some<CDenseFeatures<DataType>> features,
                  Some<CMulticlassLabels> lables,
                  const int target_dim) {
  auto isomap = some<CMultidimensionalScaling>();
  isomap->set_target_dim(target_dim);
  isomap->fit(features);

  auto new_features =
      static_cast<CDenseFeatures<DataType>*>(isomap->transform(features));

  Clusters clusters;
  auto feature_matrix = new_features->get_feature_matrix();
  for (index_t i = 0; i < new_features->get_num_vectors(); ++i) {
    auto new_vector = feature_matrix.get_column(i);
    auto label = static_cast<int>(lables->get_label(i));
    clusters[label].first.push_back(new_vector[0]);
    clusters[label].second.push_back(new_vector[1]);
  }

  PlotClusters(clusters, "Multidimensional Scaling", "mds-shogun.png");
}

void IsomapReduction(Some<CDenseFeatures<DataType>> features,
                     Some<CMulticlassLabels> lables,
                     const int target_dim) {
  auto isomap = some<CIsomap>();
  isomap->set_target_dim(target_dim);
  isomap->set_k(100);
  isomap->fit(features);

  auto new_features =
      static_cast<CDenseFeatures<DataType>*>(isomap->transform(features));

  Clusters clusters;
  auto feature_matrix = new_features->get_feature_matrix();
  for (index_t i = 0; i < new_features->get_num_vectors(); ++i) {
    auto new_vector = feature_matrix.get_column(i);
    auto label = static_cast<int>(lables->get_label(i));
    clusters[label].first.push_back(new_vector[0]);
    clusters[label].second.push_back(new_vector[1]);
  }

  PlotClusters(clusters, "Isomap", "isomap-shogun.png");
}

void KernelPCAReduction(Some<CDenseFeatures<DataType>> features,
                        Some<CMulticlassLabels> lables,
                        const int target_dim) {
  auto gauss_kernel = some<CGaussianKernel>(features, features, 0.5);
  auto pca = some<CKernelPCA>();
  pca->set_kernel(gauss_kernel.get());
  pca->set_target_dim(target_dim);
  pca->fit(features);

  Clusters clusters;
  auto feature_matrix = features->get_feature_matrix();
  for (index_t i = 0; i < features->get_num_vectors(); ++i) {
    auto vector = feature_matrix.get_column(i);
    auto new_vector = pca->apply_to_feature_vector(vector);
    auto label = static_cast<int>(lables->get_label(i));
    clusters[label].first.push_back(new_vector[0]);
    clusters[label].second.push_back(new_vector[1]);
  }

  PlotClusters(clusters, "Kernel PCA", "kernel-pca-shogun.png");
}

void PCAReduction(Some<CDenseFeatures<DataType>> features,
                  Some<CMulticlassLabels> lables,
                  const int target_dim) {
  auto pca = some<CPCA>();
  pca->set_target_dim(target_dim);
  pca->fit(features);

  Clusters clusters;
  auto feature_matrix = features->get_feature_matrix();
  for (index_t i = 0; i < features->get_num_vectors(); ++i) {
    auto vector = feature_matrix.get_column(i);
    auto new_vector = pca->apply_to_feature_vector(vector);
    auto label = static_cast<int>(lables->get_label(i));
    clusters[label].first.push_back(new_vector[0]);
    clusters[label].second.push_back(new_vector[1]);
  }

  PlotClusters(clusters, "PCA", "pca-shogun.png");
}

int main(int argc, char** argv) {
  init_shogun_with_defaults();
  if (argc > 1) {
    auto data_dir = fs::path(argv[1]);
    auto data_file_path = data_dir / data_file_name;
    auto lables_file_path = data_dir / labels_file_name;
    if (fs::exists(data_file_path) && fs::exists(lables_file_path)) {
      auto data_file = some<CCSVFile>(data_file_path.string().c_str());
      data_file->set_delimiter(' ');
      Matrix data;
      data.load(data_file);

      auto labels_file = some<CCSVFile>(lables_file_path.string().c_str());
      Matrix lables;
      lables.load(labels_file);

      // create a dataset
      auto features = some<CDenseFeatures<DataType>>(data);
      auto cluster_labels = some<CMulticlassLabels>(lables.get_row_vector(0));

      Plot3DData(features, cluster_labels);

      int target_dim = 2;
      PCAReduction(features, cluster_labels, target_dim);
      KernelPCAReduction(features, cluster_labels, target_dim);
      IsomapReduction(features, cluster_labels, target_dim);
      MDSReduction(features, cluster_labels, target_dim);
      ICAReduction(features, cluster_labels, target_dim);
      FAReduction(features, cluster_labels, target_dim);
      TSNEReduction(features, cluster_labels, target_dim);
    } else {
      std::cerr << "Dataset file " << data_file_path << " missed\n";
    }

  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }

  exit_shogun();
  return 0;
}
