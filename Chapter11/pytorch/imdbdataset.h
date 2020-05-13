#ifndef MNISTDATASET_H
#define MNISTDATASET_H

#include "imdbreader.h"
#include "vocabulary.h"

#include <torch/torch.h>

#include <string>

using ImdbData = std::pair<torch::Tensor, torch::Tensor>;
using ImdbExample = torch::data::Example<ImdbData, torch::Tensor>;

class ImdbDataset : public torch::data::Dataset<ImdbDataset, ImdbExample> {
 public:
  ImdbDataset(ImdbReader* reader,
              Vocabulary* vocabulary,
              torch::DeviceType device);

  // torch::data::Dataset implementation
  ImdbExample get(size_t index) override;
  torch::optional<size_t> size() const override;

 private:
  torch::DeviceType device_{torch::DeviceType::CPU};
  ImdbReader* reader_{nullptr};
  Vocabulary* vocabulary_{nullptr};
};

#endif  // MNISTDATASET_H
