#ifndef GLOVEDICT_H
#define GLOVEDICT_H

#include <torch/torch.h>

#include <string>
#include <unordered_map>

class GloveDict {
 public:
  GloveDict(const std::string& file_name, int64_t vec_size);
  torch::Tensor Get(const std::string& key) const;
  torch::Tensor GetUnknown() const;

 private:
  torch::Tensor unknown_;
  std::unordered_map<std::string, torch::Tensor> dict_;
};

#endif  // GLOVEDICT_H
