#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/io/File.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/util/factory.h>

#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <regex>

namespace fs = std::experimental::filesystem;

using namespace shogun;
using DataType = float64_t;
using Matrix = shogun::SGMatrix<DataType>;

int main(int argc, char** argv) {
  shogun::init_shogun_with_defaults();
  if (argc > 1 && fs::exists(argv[1])) {
    // we need to convert label to numbers to read whole file with shogun
    // functions
    {
      std::ifstream data_stream(argv[1]);
      std::string data_string((std::istreambuf_iterator<char>(data_stream)),
                              std::istreambuf_iterator<char>());

      // replace string labels, because SharkML parssr can't handle strings
      data_string =
          std::regex_replace(data_string, std::regex("Iris-setosa"), "1");
      data_string =
          std::regex_replace(data_string, std::regex("Iris-versicolor"), "2");
      data_string =
          std::regex_replace(data_string, std::regex("Iris-virginica"), "3");
      std::ofstream out_stream("iris_fix.csv");
      out_stream << data_string;
    }

    auto csv_file = shogun::some<shogun::CCSVFile>("iris_fix.csv");
    Matrix data;
    data.load(csv_file);

    // Exclude classification info from data
    // Shogun csv loader loads matrixes in column major order
    Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);
    Matrix inputs = data.submatrix(0, data.num_cols - 1);  // make a view
    inputs = inputs.clone();                               // copy exact data
    Matrix outputs = data.submatrix(4, 5);                 // make a view
    outputs = outputs.clone();                             // copy exact data
    // Transpose back because shogun algorithms expect that samples are in
    // colums
    Matrix::transpose_matrix(inputs.matrix, inputs.num_rows, inputs.num_cols);

    // create a dataset
    auto features = shogun::some<shogun::CDenseFeatures<DataType>>(inputs);

    std::cout << "samples num = " << features->get_num_vectors() << "\n"
              << "features num = " << features->get_num_features() << std::endl;

    auto features_matrix = features->get_feature_matrix();
    // Show first 5 samples
    for (int i = 0; i < 5; ++i) {
      std::cout << "Sample idx " << i << " ";
      features_matrix.get_column(i).display_vector();
    }

    auto labels =
        shogun::wrap(new shogun::CMulticlassLabels(outputs.get_column(0)));

    std::cout << "labels num = " << labels->get_num_labels() << std::endl;

    std::cout << "Label idx 0 = " << labels->get_label(0) << std::endl;
    std::cout << "Label idx 50 = " << labels->get_label(50) << std::endl;
    std::cout << "Label idx 100 = " << labels->get_label(100) << std::endl;

    // feature scaling
    auto scaler = shogun::wrap(new shogun::CRescaleFeatures());
    scaler->fit(features);
    scaler->transform(features);
    // Show rescaled samples
    for (int i = 0; i < 5; ++i) {
      std::cout << "Sample idx " << i << " ";
      features_matrix.get_column(i).display_vector();
    }
  }

  shogun::exit_shogun();
  return 0;
}
