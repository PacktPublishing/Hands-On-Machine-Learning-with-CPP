#ifndef LENET5_H
#define LENET5_H

#include <torch/torch.h>

class LeNet5Impl : public torch::nn::Module {
 public:
  LeNet5Impl();

  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Sequential conv_;
  torch::nn::Sequential full_;
};

TORCH_MODULE(LeNet5);

#endif  // LENET5_H
