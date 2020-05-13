#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Normalizer.h>

#include <experimental/filesystem>
#include <iostream>
#include <regex>
namespace fs = std::experimental::filesystem;

using namespace shark;

int main(int argc, char** argv) {
  try {
    if (argc > 1) {
      if (fs::exists(argv[1])) {
        // we need to preprocess dataset because SharkML fails to load csv with
        // string values
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

        ClassificationDataset dataset;
        csvStringToData(dataset, data_string, LAST_COLUMN);
        dataset.shuffle();
        std::size_t classes = numberOfClasses(dataset);
        std::cout << "Number of classes " << classes << std::endl;
        std::vector<std::size_t> sizes = classSizes(dataset);
        std::cout << "Class size: " << std::endl;
        for (auto cs : sizes) {
          std::cout << cs << std::endl;
        }
        std::size_t dim = inputDimension(dataset);
        std::cout << "Input dimension " << dim << std::endl;

        // normalization
        Normalizer<RealVector> normalizer;
        NormalizeComponentsUnitVariance<RealVector> normalizingTrainer(
            /*removeMean*/ true);
        normalizingTrainer.train(normalizer, dataset.inputs());
        dataset = transformInputs(dataset, normalizer);
        std::cout << dataset << std::endl;

      } else {
        std::cerr << "Invalid file path " << argv[1] << "\n";
      }
    } else {
      std::cerr << "Please specify path to the dataset\n";
    }
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }

  return 0;
}
