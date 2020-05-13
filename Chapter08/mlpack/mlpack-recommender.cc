#include "../eigen/data_loader.h"

#include <experimental/filesystem>
#include <iostream>

#include <mlpack/core.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/cf/cf.hpp>
#include <mlpack/methods/cf/decomposition_policies/batch_svd_method.hpp>

namespace fs = std::experimental::filesystem;
using DataType = double;

int main(int argc, char** argv) {
  if (argc > 1) {
    mlpack::Log::Info.ignoreInput = false;
    mlpack::Log::Warn.ignoreInput = false;
    // mlpack::Log::Debug.ignoreInput = false;
    mlpack::Log::Fatal.ignoreInput = false;

    auto root_path = fs::path(argv[1]);

    std::cout << "Data loading .." << std::endl;

    auto movies_file = root_path / "movies.csv";
    auto movies = LoadMovies(movies_file);

    auto ratings_file = root_path / "ratings.csv";
    auto ratings = LoadRatings(ratings_file);

    std::cout << "Data loaded" << std::endl;

    // The data which the CF constructor takes should be an Armadillo matrix
    // (arma::mat ) with three rows. The first row corresponds to users; the
    // second row corresponds to items; the third column corresponds to the
    // rating. This is a coordinate list format. Or a sparse matrix representing
    // (user, item) table

    arma::SpMat<DataType> ratings_matrix(ratings.size(), movies.size());
    std::vector<std::string> movie_titles;
    {
      // merge movies and users
      std::cout << "Data merging..." << std::endl;
      // fill matrix
      movie_titles.resize(movies.size());

      size_t user_idx = 0;
      for (auto& r : ratings) {
        for (auto& m : r.second) {
          auto mi = movies.find(m.first);
          auto movie_idx = std::distance(movies.begin(), mi);
          movie_titles[static_cast<size_t>(movie_idx)] = mi->second;
          ratings_matrix(user_idx, movie_idx) = static_cast<DataType>(m.second);
        }
        ++user_idx;
      }
      std::cout << "Data merged" << std::endl;
    }

    // factorization rank
    size_t n_factors = 100;

    // mlpack to avoid calculating the full rating matrix,  do
    // nearest neighbor search only on the H matrix, using the observation that
    // if the rating matrix X = W*H, then d(X.col(i), X.col(j)) = d(W H.col(i),
    // W H.col(j)).  This can be seen as nearest neighbor search on the H
    // matrix with the Mahalanobis distance.
    size_t neighborhood = 50;

    // Non negative matrix factorization  with Alternating Least Squares
    // approach
    mlpack::cf::NMFPolicy decomposition_policy;

    // mlpack::cf::BatchSVDPolicy decomposition_policy;

    // stoping criterions
    size_t max_iterations = 20;
    double min_residue = 1e-3;

    std::cout << "Training..." << std::endl;
    mlpack::cf::CFType cf(ratings_matrix, decomposition_policy, neighborhood,
                          n_factors, max_iterations, min_residue);

    std::cout << "Training done" << std::endl;

    std::cout << "Predicting..." << std::endl;
    arma::Mat<size_t> recommendations;
    // Get 5 recommendations for specified users.
    arma::Col<size_t> users;
    users << 1 << 2 << 3;

    cf.GetRecommendations(5, recommendations, users);
    std::cout << "Predicting done" << std::endl;

    for (size_t u = 0; u < recommendations.n_cols; ++u) {
      std::cout << "User " << users(u) << " recomendations are: ";
      for (size_t i = 0; i < recommendations.n_rows; ++i) {
        std::cout << movie_titles[recommendations(i, u)] << ";";
      }
      std::cout << std::endl;
    }
  } else {
    std::cerr << "Please provider path to the dataset folder\n";
  }

  return 0;
}
