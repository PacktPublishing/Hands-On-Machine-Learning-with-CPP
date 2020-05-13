#include <csv.h>
#include <Eigen/Dense>

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

template <std::size_t... Idx, typename T, typename R>
bool read_row_help(std::index_sequence<Idx...>, T& row, R& r) {
  return r.read_row(std::get<Idx>(row)...);
}

template <std::size_t... Idx, typename T>
void fill_values(std::index_sequence<Idx...>,
                 T& row,
                 std::vector<double>& data) {
  data.insert(data.end(), {std::get<Idx>(row)...});
}

int main(int argc, char** argv) {
  if (argc > 1) {
    auto file_path = fs::path(argv[1]);
    if (fs::exists(file_path)) {
      const uint32_t columns_num = 5;
      io::CSVReader<columns_num> csv_reader(file_path);

      std::vector<std::string> categorical_column;
      std::vector<double> values;
      using RowType = std::tuple<double, double, double, double, std::string>;
      RowType row;

      uint32_t rows_num = 0;
      try {
        bool done = false;
        while (!done) {
          done = !read_row_help(
              std::make_index_sequence<std::tuple_size<RowType>::value>{}, row,
              csv_reader);
          if (!done) {
            categorical_column.push_back(std::get<4>(row));
            fill_values(std::make_index_sequence<columns_num - 1>{}, row,
                        values);
            ++rows_num;
          }
        }
      } catch (const io::error::no_digit& err) {
        // ignore bad formated samples
        std::cerr << err.what() << std::endl;
      }

      auto x_data = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>(
          values.data(), rows_num, columns_num - 1);

      std::cout << x_data << std::endl;

      // Feature-scaling(Normalization):
      // Standardization - zero mean + 1 std
      Eigen::Array<double, 1, Eigen::Dynamic> std_dev =
          ((x_data.rowwise() - x_data.colwise().mean())
               .array()
               .square()
               .colwise()
               .sum() /
           (x_data.rows() - 1))
              .sqrt();

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_data_std =
          (x_data.rowwise() - x_data.colwise().mean()).array().rowwise() /
          std_dev;

      std::cout << x_data_std << std::endl;

      // Min-Max normalization
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_data_min_max =
          (x_data.rowwise() - x_data.colwise().minCoeff()).array().rowwise() /
          (x_data.colwise().maxCoeff() - x_data.colwise().minCoeff()).array();

      std::cout << x_data_min_max << std::endl;

      // Average normalization
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_data_avg =
          (x_data.rowwise() - x_data.colwise().mean()).array().rowwise() /
          (x_data.colwise().maxCoeff() - x_data.colwise().minCoeff()).array();

      std::cout << x_data_avg << std::endl;

    } else {
      std::cout << "File path is incorrect " << file_path << "\n";
    }
  } else {
    std::cout << "Please provide a path to a dataset file\n";
  }

  return 0;
}
