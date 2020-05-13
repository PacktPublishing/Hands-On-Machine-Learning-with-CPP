#include "mnistdataset.h"
#include <cassert>
#include <fstream>

namespace {
template <class T>
bool read_header(T* out, std::istream& stream) {
  auto size = static_cast<std::streamsize>(sizeof(T));
  T value;
  if (!stream.read(reinterpret_cast<char*>(&value), size)) {
    return false;
  } else {
    // flip endianness
    *out = (value << 24) | ((value << 8) & 0x00FF0000) |
           ((value >> 8) & 0X0000FF00) | (value >> 24);
    return true;
  }
}

torch::Tensor CvImageToTensor(const cv::Mat& image, torch::DeviceType device) {
  assert(image.channels() == 1);

  std::vector<int64_t> dims{static_cast<int64_t>(1),
                            static_cast<int64_t>(image.rows),
                            static_cast<int64_t>(image.cols)};

  torch::Tensor tensor_image =
      torch::from_blob(
          image.data, torch::IntArrayRef(dims),
          torch::TensorOptions().dtype(torch::kFloat).requires_grad(false))
          .clone();  // clone is required to copy data from temporary object
  return tensor_image.to(device);
}

}  // namespace

void MNISTDataset::ReadLabels(const std::string& labels_file_name) {
  std::ifstream labels_file(labels_file_name,
                            std::ios::binary | std::ios::binary);
  labels_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  if (labels_file) {
    uint32_t magic_num = 0;
    uint32_t num_items = 0;
    if (read_header(&magic_num, labels_file) &&
        read_header(&num_items, labels_file)) {
      labels_.resize(static_cast<size_t>(num_items));
      labels_file.read(reinterpret_cast<char*>(labels_.data()), num_items);
    }
  }
}

void MNISTDataset::ReadImages(const std::string& images_file_name) {
  std::ifstream labels_file(images_file_name,
                            std::ios::binary | std::ios::binary);
  labels_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  if (labels_file) {
    uint32_t magic_num = 0;
    uint32_t num_items = 0;
    rows_ = 0;
    columns_ = 0;
    if (read_header(&magic_num, labels_file) &&
        read_header(&num_items, labels_file) &&
        read_header(&rows_, labels_file) &&
        read_header(&columns_, labels_file)) {
      assert(num_items == labels_.size());
      images_.resize(num_items);
      cv::Mat img(static_cast<int>(rows_), static_cast<int>(columns_), CV_8UC1);

      for (uint32_t i = 0; i < num_items; ++i) {
        labels_file.read(reinterpret_cast<char*>(img.data),
                         static_cast<std::streamsize>(img.size().area()));
        img.convertTo(images_[i], CV_32F);
        images_[i] /= 255;  // normalize
        cv::resize(images_[i], images_[i],
                   cv::Size(32, 32));  // Resize to 32x32 size
      }
    }
  }
}

MNISTDataset::MNISTDataset(const std::string& images_file_name,
                           const std::string& labels_file_name,
                           torch::DeviceType device)
  : device_(device) {
  ReadLabels(labels_file_name);
  ReadImages(images_file_name);
}

void MNISTDataset::ShowItem(size_t index) const {
  cv::imshow(std::to_string(labels_[index]), images_[index]);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

torch::data::Example<> MNISTDataset::get(size_t index) {
  return {CvImageToTensor(images_[index], device_),
          torch::tensor(static_cast<int64_t>(labels_[index]),
                        torch::TensorOptions()
                            .dtype(torch::kLong)
                            .device(device_))};
}

torch::optional<size_t> MNISTDataset::size() const {
  return labels_.size();
}
