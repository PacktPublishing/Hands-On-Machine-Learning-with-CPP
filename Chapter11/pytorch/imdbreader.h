#ifndef IMDBREADER_H
#define IMDBREADER_H

#include <string>
#include <unordered_map>
#include <vector>

class ImdbReader {
 public:
  ImdbReader(const std::string& root_path);

  size_t GetPosSize() const;
  size_t GetNegSize() const;
  size_t GetMaxSize() const;

  using Review = std::vector<std::string>;
  const Review& GetPos(size_t index) const;
  const Review& GetNeg(size_t index) const;

 private:
  using Reviews = std::vector<Review>;

  void ReadDirectory(const std::string& path, Reviews& reviews);

 private:
  Reviews pos_samples_;
  Reviews neg_samples_;
  size_t max_size_{0};
};

using WordsFrequencies = std::unordered_map<std::string, size_t>;
void GetWordsFrequencies(const ImdbReader& reader,
                         WordsFrequencies& frequencies);

void SelectTopFrequencies(WordsFrequencies& vocab, int64_t new_size);
#endif  // IMDBREADER_H
