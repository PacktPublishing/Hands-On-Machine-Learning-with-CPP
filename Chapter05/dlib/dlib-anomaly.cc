#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <plot.h>

#include "isolation-forest.h"

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

void MultivariateGaussianDist(const Matrix& normal,
                              const Matrix& test,
                              const std::string& file_name) {
  // assume that rows are samples and columns are features

  // calculate per feature mean
  dlib::matrix<double> mu(1, normal.nc());
  dlib::set_all_elements(mu, 0);

  for (long c = 0; c < normal.nc(); ++c) {
    auto col_mean = dlib::mean(dlib::colm(normal, c));
    dlib::set_colm(mu, c) = col_mean;
  }

  // calculate covariance matrix
  dlib::matrix<double> cov(normal.nc(), normal.nc());
  dlib::set_all_elements(cov, 0);
  for (long r = 0; r < normal.nr(); ++r) {
    auto row = dlib::rowm(normal, r);
    cov += dlib::trans(row - mu) * (row - mu);
  }
  cov *= 1.0 / normal.nr();
  double cov_det = dlib::det(cov);
  dlib::matrix<double> cov_inv = dlib::inv(cov);

  auto first_part =
      1. / std::pow(2. * M_PI, normal.nc() / 2.) / std::sqrt(cov_det);

  // define probability function
  auto prob = [&](const dlib::matrix<double>& sample) {
    dlib::matrix<double> s = sample - mu;
    dlib::matrix<double> exp_val_m = s * (cov_inv * dlib::trans(s));
    double exp_val = -0.5 * exp_val_m(0, 0);
    double p = first_part * std::exp(exp_val);
    return p;
  };

  Clusters clusters;  // there will two clusters with normal and anomaly data

  // change this parameter to see descision boundary
  double prob_threshold = 0.001;

  auto detect = [&](auto samples) {
    for (long r = 0; r < samples.nr(); ++r) {
      auto row = dlib::rowm(samples, r);
      double x = row(0, 0);
      double y = row(0, 1);
      auto p = prob(row);
      if (p >= prob_threshold) {
        clusters[0].first.push_back(x);
        clusters[0].second.push_back(y);
      } else {
        clusters[1].first.push_back(x);
        clusters[1].second.push_back(y);
      }
    }
  };

  detect(normal);
  detect(test);
  PlotClusters(clusters, "Multivariate Gaussian Distribution", file_name);
}

void OneClassSvm(const Matrix& normal,
                 const Matrix& test,
                 const std::string& file_name) {
  typedef matrix<double, 0, 1> sample_type;
  typedef radial_basis_kernel<sample_type> kernel_type;
  svm_one_class_trainer<kernel_type> trainer;
  trainer.set_nu(0.5);                   // control smoothness of the solution
  trainer.set_kernel(kernel_type(0.5));  // kernel bandwidth
  std::vector<sample_type> samples;
  for (long r = 0; r < normal.nr(); ++r) {
    auto row = rowm(normal, r);
    samples.push_back(row);
  }
  decision_function<kernel_type> df = trainer.train(samples);
  Clusters clusters;
  double threshold = -2.0;

  auto detect = [&](auto samples) {
    for (long r = 0; r < samples.nr(); ++r) {
      auto row = dlib::rowm(samples, r);
      double x = row(0, 0);
      double y = row(0, 1);
      auto p = df(row);
      if (p > threshold) {
        clusters[0].first.push_back(x);
        clusters[0].second.push_back(y);
      } else {
        clusters[1].first.push_back(x);
        clusters[1].second.push_back(y);
      }
    }
  };

  detect(normal);
  detect(test);
  PlotClusters(clusters, "One Class SVM", file_name);
}

void IsolationForest(const Matrix& normal,
                     const Matrix& test,
                     const std::string& file_name) {
  iforest::Dataset<2> dataset;

  auto put_to_dataset = [&](const Matrix& samples) {
    for (long r = 0; r < samples.nr(); ++r) {
      auto row = dlib::rowm(samples, r);
      double x = row(0, 0);
      double y = row(0, 1);
      dataset.push_back({x, y});
    }
  };

  put_to_dataset(normal);
  put_to_dataset(test);

  iforest::IsolationForest iforest(dataset, 300, 50);

  Clusters clusters;
  double threshold = 0.6;  // change this value to see isolation boundary
  for (auto& s : dataset) {
    auto anomaly_score = iforest.AnomalyScore(s);
    // std::cout << anomaly_score << " " << s[0] << " " << s[1] << std::endl;

    if (anomaly_score < threshold) {
      clusters[0].first.push_back(s[0]);
      clusters[0].second.push_back(s[1]);
    } else {  // anomaly
      clusters[1].first.push_back(s[0]);
      clusters[1].second.push_back(s[1]);
    }
  }

  PlotClusters(clusters, "Isolation Forest", file_name);
}

using Dataset = std::pair<Matrix, Matrix>;

Dataset LoadDataset(const fs::path& file_path) {
  if (fs::exists(file_path)) {
    std::ifstream file(file_path);
    matrix<DataType> data;
    file >> data;

    long n_normal = 50;
    Matrix normal =
        dlib::subm(data, range(0, n_normal - 1), range(0, data.nc() - 1));
    Matrix test = dlib::subm(data, range(n_normal, data.nr() - 1),
                             range(0, data.nc() - 1));
    return {normal, test};
  } else {
    std::string msg = "Dataset file " + file_path.string() + " missed\n";
    throw std::invalid_argument(msg);
  }
}

Dataset CombineDatasets(const Dataset& a, const Dataset& b) {
  Matrix normal(a.first.nr() + b.first.nr(), a.first.nc());
  set_subm(normal, range(0, a.first.nr() - 1), range(0, a.first.nc() - 1)) =
      a.first;
  set_subm(normal, range(a.first.nr(), normal.nr() - 1),
           range(0, a.first.nc() - 1)) = b.first;

  Matrix test(a.second.nr() + b.second.nr(), a.second.nc());
  set_subm(test, range(0, a.second.nr() - 1), range(0, a.second.nc() - 1)) =
      a.second;
  set_subm(test, range(a.second.nr(), test.nr() - 1),
           range(0, a.second.nc() - 1)) = b.second;

  return {normal, test};
}

int main(int argc, char** argv) {
  if (argc > 1) {
    try {
      auto base_dir = fs::path(argv[1]);

      std::string data_name_multi{"multivar.csv"};
      std::string data_name_uni{"univar.csv"};

      auto dataset_multi = LoadDataset(base_dir / data_name_multi);
      auto dataset_uni = LoadDataset(base_dir / data_name_uni);

      MultivariateGaussianDist(dataset_multi.first, dataset_multi.second,
                               "dlib-multi-var.png");
      OneClassSvm(dataset_multi.first, dataset_multi.second, "dlib-ocsvm.png");

      // make dataset with two clusters

      auto dataset_combi = CombineDatasets(dataset_multi, dataset_uni);
      OneClassSvm(dataset_combi.first, dataset_combi.second,
                  "dlib-ocsvm_two.png");

      IsolationForest(dataset_combi.first, dataset_combi.second,
                      "dlib-iforest-two.png");

    } catch (const std::exception& err) {
      std::cerr << err.what();
    }
  } else {
    std::cerr << "Please provider path to the datasets folder\n";
  }
  return 0;
}
