#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/statistics.h>
#include <plot.h>

#include <experimental/filesystem>
#include <iostream>
#include <unordered_map>

using namespace dlib;
namespace fs = std::experimental::filesystem;

const std::vector<std::string> colors{"black", "red",    "blue",  "green",
                                      "cyan",  "yellow", "brown", "magenta"};

using DataType = double;
using Matrix = matrix<DataType>;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<size_t, PointCoords>;

const std::string photo_file_name{"photo.png"};
const std::string data_file_name{"swissroll.dat"};
const std::string labels_file_name{"swissroll_labels.dat"};

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

void SammonReduction(const std::vector<Matrix>& data,
                     const std::vector<unsigned long>& labels,
                     long target_dim) {
  dlib::sammon_projection sp;
  auto new_data = sp(data, target_dim);

  Clusters clusters;
  for (size_t r = 0; r < new_data.size(); ++r) {
    Matrix vec = new_data[r];
    double x = vec(0, 0);
    double y = vec(1, 0);
    auto l = labels[r];
    clusters[l].first.push_back(x);
    clusters[l].second.push_back(y);
  }

  PlotClusters(clusters, "Sammon Mapping", "sammon-dlib.png");
}

void PCAReduction(const std::vector<Matrix>& data,
                  const std::vector<unsigned long>& labels,
                  double target_dim) {
  dlib::vector_normalizer_pca<Matrix> pca;
  pca.train(data, target_dim / data[0].nr());
  std::vector<Matrix> new_data;
  new_data.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    new_data.emplace_back(pca(data[i]));
  }

  Clusters clusters;
  for (size_t r = 0; r < new_data.size(); ++r) {
    Matrix vec = new_data[r];
    double x = vec(0, 0);
    double y = vec(1, 0);
    auto l = labels[r];
    clusters[l].first.push_back(x);
    clusters[l].second.push_back(y);
  }

  PlotClusters(clusters, "PCA", "pca-dlib.png");
}

void LDAReduction(const Matrix& data,
                  const std::vector<unsigned long>& labels,
                  unsigned long target_dim) {
  dlib::matrix<DataType, 0, 1> mean;
  Matrix transform = data;
  dlib::compute_lda_transform(transform, mean, labels, target_dim);

  Clusters clusters;
  for (long r = 0; r < data.nr(); ++r) {
    Matrix row = transform * dlib::trans(dlib::rowm(data, r)) - mean;
    double x = row(0, 0);
    double y = row(1, 0);
    auto l = labels[static_cast<size_t>(r)];
    clusters[l].first.push_back(x);
    clusters[l].second.push_back(y);
  }

  PlotClusters(clusters, "LDA", "lda-dlib.png");
}

void PCACompression(const std::string& image_file, long target_dim) {
  array2d<dlib::rgb_pixel> img;
  load_image(img, image_file);

  array2d<unsigned char> img_gray;
  assign_image(img_gray, img);
  save_png(img_gray, "original.png");

  array2d<DataType> tmp;
  assign_image(tmp, img_gray);
  Matrix img_mat = dlib::mat(tmp);
  img_mat /= 255.;  // scale

  std::cout << "Original data size " << img_mat.size() << std::endl;

  // take patches 8x8
  std::vector<Matrix> data;
  int patch_size = 8;

  for (long r = 0; r < img_mat.nr(); r += patch_size) {
    for (long c = 0; c < img_mat.nc(); c += patch_size) {
      auto sm = dlib::subm(img_mat, r, c, patch_size, patch_size);
      data.emplace_back(dlib::reshape_to_column_vector(sm));
    }
  }

  // normalize data
  auto data_mat = mat(data);
  Matrix m = mean(data_mat);
  Matrix sd = reciprocal(sqrt(variance(data_mat)));

  matrix<decltype(data_mat)::type, 0, 1, decltype(data_mat)::mem_manager_type>
      x(data_mat);
  for (long r = 0; r < x.size(); ++r)
    x(r) = pointwise_multiply(x(r) - m, sd);

  // perform PCA
  Matrix temp, eigen, pca;
  // Compute the svd of the covariance matrix
  dlib::svd(covariance(x), temp, eigen, pca);
  Matrix eigenvalues = diag(eigen);

  rsort_columns(pca, eigenvalues);

  // leave only required number of principal components
  pca = trans(colm(pca, range(0, target_dim)));

  // dimensionality reduction
  std::vector<Matrix> new_data;
  size_t new_size = 0;
  new_data.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    new_data.emplace_back(pca * data[i]);
    new_size += static_cast<size_t>(new_data.back().size());
  }

  std::cout << "New data size " << new_size + static_cast<size_t>(pca.size())
            << std::endl;

  // unpack data
  auto pca_matrix_t = dlib::trans(pca);
  Matrix isd = dlib::reciprocal(sd);
  for (size_t i = 0; i < new_data.size(); ++i) {
    Matrix sample = pca_matrix_t * new_data[i];
    new_data[i] = dlib::pointwise_multiply(sample, isd) + m;
  }

  size_t i = 0;
  for (long r = 0; r < img_mat.nr(); r += patch_size) {
    for (long c = 0; c < img_mat.nc(); c += patch_size) {
      auto sm = dlib::reshape(new_data[i], patch_size, patch_size);
      dlib::set_subm(img_mat, r, c, patch_size, patch_size) = sm;
      ++i;
    }
  }

  img_mat *= 255.0;
  assign_image(img_gray, img_mat);
  equalize_histogram(img_gray);
  save_png(img_gray, "compressed.png");
}

int main(int argc, char** argv) {
  if (argc > 1) {
    try {
      auto data_dir = fs::path(argv[1]);
      auto data_file_path = data_dir / data_file_name;
      auto lables_file_path = data_dir / labels_file_name;
      auto photo_file_path = data_dir / photo_file_name;
      if (fs::exists(data_file_path) && fs::exists(lables_file_path) &&
          fs::exists(photo_file_path)) {
        matrix<DataType> data;
        std::vector<Matrix> vdata;
        {
          std::ifstream file(data_file_path);
          file >> data;
          vdata.reserve(static_cast<size_t>(data.nr()));
          for (long row = 0; row < data.nr(); ++row) {
            vdata.emplace_back(dlib::reshape_to_column_vector(
                dlib::subm_clipped(data, row, 0, 1, data.nc())));
          }
        }
        matrix<DataType> labels;
        std::vector<unsigned long> vlables;
        {
          std::ifstream file(lables_file_path);
          file >> labels;
          vlables.resize(static_cast<size_t>(labels.nr()));
          for (long r = 0; r < labels.nr(); ++r) {
            vlables[static_cast<size_t>(r)] =
                static_cast<unsigned long>(labels(r, 0));
          }
        }

        PCACompression(photo_file_path, 10);
        // int target_dim = 2;
        // LDAReduction(data, vlables, target_dim);
        // PCAReduction(vdata, vlables, target_dim);
        // SammonReduction(vdata, vlables, target_dim);
      }
    } catch (const std::exception& err) {
      std::cerr << err.what();
    }
  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }
  return 0;
}
