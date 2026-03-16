/*
PR's that introduced breaking changes:
  https://github.com/pytorch/pytorch/pull/34322

W_ih = torch_rnn.weight_ih_l0.detach()
b_ih = torch_rnn.bias_ih_l0.detach()
W_hh = torch_rnn.weight_hh_l0.detach()
b_hh = torch_rnn.bias_hh_l0.detach()
*/
#ifndef LENET5_H
#define LENET5_H

#include <torch/torch.h>

class PackedLSTMImpl : public torch::nn::Module {
 public:
  explicit PackedLSTMImpl(const torch::nn::LSTMOptions& options);

  std::vector<torch::Tensor> flat_weights() const;

  std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input,
                               const torch::Tensor& lengths,
                               torch::Tensor state = {});

  const torch::nn::LSTMOptions& options() const;

 private:
  torch::nn::LSTM rnn_ = nullptr;
};

TORCH_MODULE(PackedLSTM);

class SentimentRNNImpl : public torch::nn::Module {
 public:
  SentimentRNNImpl(int64_t vocab_size,
                   int64_t embedding_dim,
                   int64_t hidden_dim,
                   int64_t output_dim,
                   int64_t n_layers,
                   bool bidirectional,
                   double dropout,
                   int64_t pad_idx);

  void SetPretrainedEmbeddings(const torch::Tensor& weights);

  torch::Tensor forward(const torch::Tensor& text, const at::Tensor& length);

 private:
  int64_t pad_idx_{-1};
  torch::autograd::Variable embeddings_weights_;
  PackedLSTM rnn_ = nullptr;
  torch::nn::Linear fc_ = nullptr;
  torch::nn::Dropout dropout_ = nullptr;
};

TORCH_MODULE(SentimentRNN);

#endif  // LENET5_H
