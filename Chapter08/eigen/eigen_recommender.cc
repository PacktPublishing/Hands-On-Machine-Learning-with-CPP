#include <omp.h>
// #define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "data_loader.h"

namespace fs = std::experimental::filesystem;
using DataType = float;
// using Eigen::ColMajor is Eigen restriction -  todense method always returns
// matrices in ColMajor order
using Matrix =
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using SparseMatrix = Eigen::SparseMatrix<DataType, Eigen::ColMajor>;

using DiagonalMatrix =
    Eigen::DiagonalMatrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

// Initialize matrix with random values and normalize them
Matrix InitialiseMatrix(Eigen::Index rows, Eigen::Index cols) {
  Matrix mat = Matrix::Random(rows, cols).array().abs();
  auto row_sums = mat.rowwise().sum();
  mat.array().colwise() /= row_sums.array();
  return mat;
}

Matrix RatingsPredictions(const Matrix& x, const Matrix& y) {
  return x * y.transpose();
}

DataType CalculateWeightedMse(const Matrix& x,
                              const Matrix& y,
                              const SparseMatrix& p,
                              const SparseMatrix& ratings_matrix,
                              DataType alpha) {
  Matrix c(ratings_matrix);
  c.array() *= alpha;
  c.array() += 1.0;

  Matrix diff(p - RatingsPredictions(x, y));
  diff = diff.array().pow(2.f);

  Matrix weighted_diff = c.array() * diff.array();
  return weighted_diff.array().mean();
}

void PrintRecommendations(const Matrix& ratings_matrix,
                          const Matrix& ratings_matrix_pred,
                          const std::vector<std::string>& movie_titles) {
  // auto m = ratings_matrix.rows();
  auto n = ratings_matrix.cols();
  std::vector<std::string> liked;
  std::vector<std::string> recommended;
  for (Eigen::Index u = 0; u < 5; ++u) {
    for (Eigen::Index i = 0; i < n; ++i) {
      DataType orig_value = ratings_matrix(u, i);
      if (orig_value >= 3.f) {
        liked.push_back(movie_titles[static_cast<size_t>(i)]);
      }
      DataType pred_value = ratings_matrix_pred(u, i);
      if (pred_value >= 0.8f && orig_value < 1.f) {
        recommended.push_back(movie_titles[static_cast<size_t>(i)]);
      }
    }
    std::cout << "\nUser " << u << " liked :";
    for (auto& l : liked) {
      std::cout << l << "; ";
    }
    std::cout << "\nUser " << u << " recommended :";
    for (auto& r : recommended) {
      std::cout << r << "; ";
    }
    std::cout << std::endl;
    liked.clear();
    recommended.clear();
  }
}

int main(int argc, char** argv) {
  if (argc == 2) {
    Eigen::initParallel();
    auto root_path = fs::path(argv[1]);
    if (fs::exists(root_path)) {
      SparseMatrix ratings_matrix;  // user-item ratings
      SparseMatrix p;               // binary variables
      std::vector<std::string> movie_titles;
      {
        std::cout << "Data loading .." << std::endl;
        // load data
        auto movies_file = root_path / "movies.csv";
        auto movies = LoadMovies(movies_file);

        auto ratings_file = root_path / "ratings.csv";
        auto ratings = LoadRatings(ratings_file);

        std::cout << "Data loaded" << std::endl;

        // merge movies and users
        std::cout << "Data merging..." << std::endl;
        // fill matrix
        ratings_matrix.resize(static_cast<Eigen::Index>(ratings.size()),
                              static_cast<Eigen::Index>(movies.size()));
        ratings_matrix.setZero();
        p.resize(ratings_matrix.rows(), ratings_matrix.cols());
        p.setZero();

        movie_titles.resize(movies.size());

        Eigen::Index user_idx = 0;
        for (auto& r : ratings) {
          for (auto& m : r.second) {
            auto mi = movies.find(m.first);
            Eigen::Index movie_idx = std::distance(movies.begin(), mi);
            movie_titles[static_cast<size_t>(movie_idx)] = mi->second;
            ratings_matrix.insert(user_idx, movie_idx) =
                static_cast<DataType>(m.second);
            p.insert(user_idx, movie_idx) = 1.0;
          }
          ++user_idx;
        }
        ratings_matrix.makeCompressed();
        std::cout << "Data merged" << std::endl;
      }

      // prepare for learning
      auto m = ratings_matrix.rows();
      auto n = ratings_matrix.cols();

      std::cout << "Users " << m << " Movies " << n << std ::endl;

      Eigen::Index n_factors = 100;
      auto y = InitialiseMatrix(n, n_factors);
      auto x = InitialiseMatrix(m, n_factors);

      // Test initialization
      DataType alpha = 40.f;  // confidence level parameter
      auto w_mse = CalculateWeightedMse(x, y, p, ratings_matrix, alpha);
      std::cout << "Initial weighted mse " << w_mse << std::endl;

      // Precalculate regularization term
      DataType reg_lambda = 0.1f;
      SparseMatrix reg =
          (reg_lambda * Matrix::Identity(n_factors, n_factors)).sparseView();

      // Define diagonal identity terms
      SparseMatrix user_diag = -1 * Matrix::Identity(n, n).sparseView();
      SparseMatrix item_diag = -1 * Matrix::Identity(m, m).sparseView();

      // define weights
      std::cout << "Calculate weights ..." << std::endl;
      std::vector<DiagonalMatrix> user_weights(static_cast<size_t>(m));
      std::vector<DiagonalMatrix> item_weights(static_cast<size_t>(n));
      {
        Matrix weights(ratings_matrix);
        weights.array() *= alpha;
        weights.array() += 1;

        for (Eigen::Index i = 0; i < m; ++i) {
          user_weights[static_cast<size_t>(i)] = weights.row(i).asDiagonal();
        }
        for (Eigen::Index i = 0; i < n; ++i) {
          item_weights[static_cast<size_t>(i)] = weights.col(i).asDiagonal();
        }
      }

      // learning loop
      size_t n_iterations = 5;
      std::cout << "Start learning ..." << std::endl;
      // omp_set_num_threads(4);
      for (size_t k = 0; k < n_iterations; ++k) {
        auto start_time = std::chrono::steady_clock::now();
        auto yt = y.transpose();
        auto yty = yt * y;

#pragma omp parallel
        {
          Matrix diff;
          Matrix ytcuy;
          Matrix a, b, update_y;
#pragma omp for private(diff, ytcuy, a, b, update_y)
          for (size_t i = 0; i < static_cast<size_t>(m); ++i) {
            diff = user_diag;
            diff += user_weights[i];
            ytcuy = yty + yt * diff * y;
            auto p_val = p.row(static_cast<Eigen::Index>(i)).transpose();

            a = ytcuy + reg;
            b = yt * user_weights[i] * p_val;

            update_y = a.colPivHouseholderQr().solve(b);
            x.row(static_cast<Eigen::Index>(i)) = update_y.transpose();
          }
        }

        auto xt = x.transpose();
        auto xtx = xt * x;

#pragma omp parallel
        {
          Matrix diff;
          Matrix xtcux;
          Matrix a, b, update_x;
#pragma omp for private(diff, xtcux, a, b, update_x)
          for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            diff = item_diag;
            diff += item_weights[i];
            xtcux = xtx + xt * diff * x;
            auto p_val = p.col(static_cast<Eigen::Index>(i));

            a = xtcux + reg;
            b = xt * item_weights[i] * p_val;

            update_x = a.colPivHouseholderQr().solve(b);
            y.row(static_cast<Eigen::Index>(i)) = update_x.transpose();
          }
        }

        w_mse = CalculateWeightedMse(x, y, p, ratings_matrix, alpha);
        auto finish_time = std::chrono::steady_clock::now();
        double elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                finish_time - start_time)
                .count();

        std::cout << "Initeration " << k << " weighted mse " << w_mse
                  << " time " << elapsed_seconds << std::endl;
      }
      std::cout << "Learning done" << std::endl;

      PrintRecommendations(ratings_matrix, RatingsPredictions(x, y),
                           movie_titles);

      return 0;
    }
  }

  std::cout << "please specify data set directory\n";
  return 0;
};
