#ifndef VOCABULARY_H
#define VOCABULARY_H

#include "glovedict.h"
#include "imdbreader.h"

#include <torch/torch.h>
#include <unordered_map>

class Vocabulary {
 public:
  Vocabulary(const WordsFrequencies& words_frequencies,
             const GloveDict& glove_dict);

  int64_t GetIndex(const std::string& word) const;
  int64_t GetPaddingIndex() const;
  torch::Tensor GetEmbeddings() const;
  int64_t GetEmbeddingsCount() const;

 private:
  std::unordered_map<std::string, size_t> words_to_index_map_;
  std::vector<torch::Tensor> embeddings_;
  size_t unk_index_;
  size_t pad_index_;
};

#endif  // VOCABULARY_H
