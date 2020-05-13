#include "lenet5.h"

static std::vector<int64_t> k_size = {2, 2};
static std::vector<int64_t> p_size = {0, 0};
static c10::optional<int64_t> divisor_override;

LeNet5Impl::LeNet5Impl() {
  conv_ = torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5)),
      torch::nn::Functional(torch::tanh),
      torch::nn::Functional(torch::avg_pool2d,
                            /*kernel_size*/ torch::IntArrayRef(k_size),
                            /*stride*/ torch::IntArrayRef(k_size),
                            /*padding*/ torch::IntArrayRef(p_size),
                            /*ceil_mode*/ false,
                            /*count_include_pad*/ false,
                            divisor_override),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5)),
      torch::nn::Functional(torch::tanh),
      torch::nn::Functional(torch::avg_pool2d,
                            /*kernel_size*/ torch::IntArrayRef(k_size),
                            /*stride*/ torch::IntArrayRef(k_size),
                            /*padding*/ torch::IntArrayRef(p_size),
                            /*ceil_mode*/ false,
                            /*count_include_pad*/ false,
                            divisor_override),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 120, 5)),
      torch::nn::Functional(torch::tanh));
  register_module("conv", conv_);

  full_ = torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(120, 84)),
      torch::nn::Functional(torch::tanh),
      torch::nn::Linear(torch::nn::LinearOptions(84, 10)));
  register_module("full", full_);
}

torch::Tensor LeNet5Impl::forward(at::Tensor x) {
  auto output = conv_->forward(x);
  output = output.view({x.size(0), -1});
  output = full_->forward(output);
  // not all of the functions can be used with torch::nn::Functional class
  // because of same names
  output = torch::log_softmax(output, -1);
  return output;
}
