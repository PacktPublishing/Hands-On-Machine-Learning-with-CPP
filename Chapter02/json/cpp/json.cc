#include "reviewsreader.h"

#include <Eigen/Dense>

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv) {
  if (argc > 1) {
    auto file_path = fs::path(argv[1]);
    if (fs::exists(file_path)) {
      auto papers = ReadPapersReviews(file_path);
      // create matrices
      Eigen::MatrixXi x_data(papers.size(), 3);
      Eigen::MatrixXi y_data(papers.size(), 1);
      Eigen::Index i = 0;
      for (const auto& p : papers) {
        if (p.preliminary_decision == "accept")
          y_data(i, 0) = 1;
        else
          y_data(i, 0) = 0;

        if (!p.reviews.empty()) {
          int64_t confidence_avg = 0;
          int64_t evaluation_avg = 0;
          int64_t orientation_avg = 0;
          for (const auto& r : p.reviews) {
            confidence_avg += std::stoi(r.confidence);
            evaluation_avg += std::stoi(r.evaluation);
            orientation_avg += std::stoi(r.orientation);
          }
          int64_t reviews_num = static_cast<int64_t>(p.reviews.size());
          x_data(i, 0) = static_cast<int>(confidence_avg / reviews_num);
          x_data(i, 1) = static_cast<int>(evaluation_avg / reviews_num);
          x_data(i, 2) = static_cast<int>(orientation_avg / reviews_num);
        }
        ++i;
      }
      std::cout << x_data << std::endl;
      std::cout << y_data << std::endl;

    } else {
      std::cout << "File path is incorrect " << file_path << "\n";
    }
  } else {
    std::cout << "Please provide a path to a dataset file\n";
  }

  return 0;
}
