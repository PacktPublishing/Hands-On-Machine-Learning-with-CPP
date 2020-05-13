#include "utils.h"

#include <opencv2/opencv.hpp>

#include <algorithm>

Classes ReadClasses(const std::string& file_name) {
  Classes classes;
  std::ifstream file(file_name);
  if (file) {
    std::string line;
    std::string id;
    std::string label;
    std::string token;
    size_t idx = 0;
    while (std::getline(file, line)) {
      std::stringstream line_stream(line);
      auto pos = line.find_first_of(" ");
      id = line.substr(0, pos);
      label = line.substr(pos + 1);
      classes.insert({idx, label});
      ++idx;
    }
  }
  return classes;
}

torch::Tensor CvImageToTensor(cv::Mat image, int width, int height) {
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

  // create tensor
  std::vector<int64_t> dims = {1, 3, height, width};
  auto tensor = torch::from_blob(image.data, dims,
                                 torch::TensorOptions()
                                     .device(at::kCPU)
                                     .dtype(at::kFloat)
                                     .requires_grad(false));
  return tensor.clone();
}

torch::Tensor ReadMemoryImageTensor(const std::string& data,
                                    int width,
                                    int height) {
  // load image
  cv::Mat image;
  {
    std::vector<char> buf(data.begin(), data.end());
    image = cv::imdecode(buf, cv::IMREAD_COLOR);
  }
  return CvImageToTensor(image, width, height);
}
