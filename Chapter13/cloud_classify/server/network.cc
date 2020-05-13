#include "network.h"
#include <torch/script.h>

Network::Network(const std::string& snapshot_path,
                 const std::string& synset_path,
                 torch::DeviceType device_type)
    : device_type_(device_type) {
  classes_ = ReadClasses(synset_path);
  model_ = torch::jit::load(snapshot_path, device_type);
}

std::string Network::Classify(const torch::Tensor& image) {
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(image.to(device_type_));
  at::Tensor output = model_.forward(inputs).toTensor();

  auto max_result = output.squeeze().max(0);
  auto max_index = std::get<1>(max_result).item<int64_t>();
  auto max_value = std::get<0>(max_result).item<float>();

  return std::to_string(max_index) + " - " + std::to_string(max_value) + " - " +
         classes_[static_cast<size_t>(max_index)];
}
