#include "imdbdataset.h"
#include <cassert>
#include <fstream>

ImdbDataset::ImdbDataset(ImdbReader* reader,
                         Vocabulary* vocabulary,
                         torch::DeviceType device)
    : device_(device), reader_(reader), vocabulary_(vocabulary) {}

ImdbExample ImdbDataset::get(size_t index) {
  torch::Tensor target;
  const ImdbReader::Review* review{nullptr};
  if (index < reader_->GetPosSize()) {
    review = &reader_->GetPos(index);
    target = torch::tensor(
        1.f, torch::dtype(torch::kFloat).device(device_).requires_grad(false));
  } else {
    review = &reader_->GetNeg(index - reader_->GetPosSize());
    target = torch::tensor(
        0.f, torch::dtype(torch::kFloat).device(device_).requires_grad(false));
  }
  // encode text
  std::vector<int64_t> indices(reader_->GetMaxSize());
  size_t i = 0;

  for (auto& w : (*review)) {
    indices[i] = vocabulary_->GetIndex(w);
    ++i;
  }
  // pad text to same size
  for (; i < indices.size(); ++i) {
    indices[i] = vocabulary_->GetPaddingIndex();
  }

  auto data = torch::from_blob(indices.data(),
                               {static_cast<int64_t>(reader_->GetMaxSize())},
                               torch::dtype(torch::kLong).requires_grad(false));
  auto data_len =
      torch::tensor(static_cast<int64_t>(review->size()),
                    torch::dtype(torch::kLong).requires_grad(false));
  return {{data.clone().to(device_), data_len.clone()}, target.squeeze()};
}

torch::optional<size_t> ImdbDataset::size() const {
  return reader_->GetPosSize() + reader_->GetNegSize();
}
