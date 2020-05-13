#include <caffe2/core/init.h>
#include <caffe2/onnx/backend.h>
#include <caffe2/utils/proto_utils.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>

using Classes = std::map<size_t, std::string>;
Classes ReadClasses(const std::string& file_name) {
  Classes classes;
  std::ifstream file(file_name);
  if (file) {
    std::string line;
    std::string id;
    std::string label;
    std::string token;
    size_t idx = 1;
    while (std::getline(file, line)) {
      std::stringstream line_stream(line);
      size_t i = 0;
      while (std::getline(line_stream, token, ' ')) {
        switch (i) {
          case 0:
            id = token;
            break;
          case 1:
            label = token;
            break;
        }
        token.clear();
        ++i;
      }
      classes.insert({idx, label});
      ++idx;
    }
  }
  return classes;
}

caffe2::TensorCPU ReadImageTensor(const std::string& file_name,
                                  int width,
                                  int height) {
  // load image
  auto image = cv::imread(file_name, cv::IMREAD_COLOR);

  if (!image.cols || !image.rows) {
    return {};
  }

  if (image.cols != width || image.rows != height) {
    // scale image to fit
    cv::Size scaled(std::max(height * image.cols / image.rows, width),
                    std::max(height, width * image.rows / image.cols));
    cv::resize(image, image, scaled);

    // crop image to fit
    cv::Rect crop((image.cols - width) / 2, (image.rows - height) / 2, width,
                  height);
    image = image(crop);
  }

  image.convertTo(image, CV_32FC3);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  CAFFE_ENFORCE_EQ(image.channels(), 3);
  CAFFE_ENFORCE_EQ(image.rows, height);
  CAFFE_ENFORCE_EQ(image.cols, width);

  std::vector<cv::Mat> channels(3);
  cv::split(image, channels);

  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> stddev = {0.229, 0.224, 0.225};

  size_t i = 0;
  for (auto& c : channels) {
    c = ((c / 255) - mean[i]) / stddev[i];
    ++i;
  }

  cv::vconcat(channels[0], channels[1], image);
  cv::vconcat(image, channels[2], image);
  assert(image.isContinuous());

  cv::imshow("image", image);
  cv::waitKey(0);
  cv::destroyAllWindows();

  // create tensor
  std::vector<int64_t> dims = {1, 3, height, width};

  caffe2::TensorCPU tensor(dims, caffe2::DeviceType::CPU);
  std::copy_n(reinterpret_cast<float*>(image.data), image.size().area(),
              tensor.mutable_data<float>());

  return tensor;
}

int main(int argc, char** argv) {
  try {
    if (argc == 4) {
      caffe2::GlobalInit(&argc, &argv);

      auto classes = ReadClasses(argv[3]);

      onnx_torch::ModelProto model_proto;
      {
        std::ifstream file(argv[1], std::ios_base::binary);
        if (!file) {
          std::cerr << "File " << argv[1] << "can't be opened\n";
          return 1;
        }
        if (!model_proto.ParseFromIstream(&file)) {
          std::cerr << "Failed to parse onnx model\n";
          return 1;
        }
      }

      std::string model_str;
      if (model_proto.SerializeToString(&model_str)) {
        caffe2::onnx::Caffe2Backend onnx_backend;
        std::vector<caffe2::onnx::Caffe2Ops> ops;
        auto model = onnx_backend.Prepare(model_str, "CPU", ops);
        if (model != nullptr) {
          caffe2::TensorCPU image = ReadImageTensor(argv[2], 224, 224);

          std::vector<caffe2::TensorCPU> inputs;
          inputs.push_back(std::move(image));

          std::cout << "Input: " << inputs[0].DebugString() << std::endl;

          std::vector<caffe2::TensorCPU> outputs(1);

          model->Run(inputs, &outputs);

          for (auto& output : outputs) {
            std::cout << "Output: " << output.DebugString() << std::endl;
            const auto& probabilities = output.data<float>();
            std::vector<std::pair<float, int>> pairs;  // prob : class index
            for (auto i = 0; i < output.size(); i++) {
              if (probabilities[i] > 0.01f) {
                pairs.push_back(
                    std::make_pair(probabilities[i], i + 1));  // 0 - background
              }
            }
            std::sort(pairs.begin(), pairs.end());
            std::reverse(pairs.begin(), pairs.end());
            pairs.resize(std::min(5UL, pairs.size()));
            for (auto& p : pairs) {
              std::cout << "Class " << p.second << " Label "
                        << classes[static_cast<size_t>(p.second)] << " Prob "
                        << p.first << std::endl;
            }
          }
        }

        google::protobuf::ShutdownProtobufLibrary();
        return 0;
      }
    } else {
      std::cout << "Usage: <ONNX model file> <data file> <classes file>";
    }
  } catch (const std::exception& err) {
    std::cerr << err.what();
  } catch (...) {
    std::cerr << "err";
  }
  return 1;
}
