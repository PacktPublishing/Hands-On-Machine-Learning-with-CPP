#ifndef MNISTDATASET_H
#define MNISTDATASET_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <string>

class MNISTDataset : public torch::data::Dataset<MNISTDataset> {
 public:
  MNISTDataset(const std::string& images_file_name,
               const std::string& labels_file_name,
               torch::DeviceType device);

  // test only method
  void ShowItem(size_t index) const;

  // torch::data::Dataset implementation
  torch::data::Example<> get(size_t index) override;
  torch::optional<size_t> size() const override;

 private:
  void ReadLabels(const std::string& labels_file_name);
  void ReadImages(const std::string& images_file_name);

  uint32_t rows_ = 0;
  uint32_t columns_ = 0;
  std::vector<unsigned char> labels_;
  std::vector<cv::Mat> images_;
  torch::DeviceType device_;
};

#endif  // MNISTDATASET_H
