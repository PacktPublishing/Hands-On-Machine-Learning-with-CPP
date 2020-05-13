#ifndef NETWORK_H
#define NETWORK_H

#include <torch/torch.h>
#include "utils.h"

class Network {
 public:
  Network(const std::string& snapshot_path,
          const std::string& synset_path,
          torch::DeviceType device_type);
  std::string Classify(const at::Tensor& image);

 private:
  torch::DeviceType device_type_;
  Classes classes_;
  torch::jit::script::Module model_;
};

#endif  // NETWORK_H
