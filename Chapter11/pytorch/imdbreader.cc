#include "imdbreader.h"

#include <experimental/filesystem>
#include <fstream>
#include <future>
#include <regex>

namespace fs = std::experimental::filesystem;

ImdbReader::ImdbReader(const std::string& root_path) {
  auto root = fs::path(root_path);
  auto neg_path = root / "neg";
  auto pos_path = root / "pos";
  if (fs::exists(neg_path) && fs::exists(pos_path)) {
    auto neg = std::async(std::launch::async,
                          [&]() { ReadDirectory(neg_path, neg_samples_); });
    auto pos = std::async(std::launch::async,
                          [&]() { ReadDirectory(pos_path, pos_samples_); });
    neg.get();
    pos.get();
  } else {
    throw std::invalid_argument("ImdbReader incorrect path");
  }
}

const ImdbReader::Review& ImdbReader::GetPos(size_t index) const {
  return pos_samples_.at(index);
}

const ImdbReader::Review& ImdbReader::GetNeg(size_t index) const {
  return neg_samples_.at(index);
}

size_t ImdbReader::GetPosSize() const {
  return pos_samples_.size();
}

size_t ImdbReader::GetNegSize() const {
  return neg_samples_.size();
}

size_t ImdbReader::GetMaxSize() const {
  return max_size_;
}

void ImdbReader::ReadDirectory(const std::string& path, Reviews& reviews) {
  std::regex re("[^a-zA-Z0-9]");
  std::sregex_token_iterator end;

  for (auto& entry : fs::directory_iterator(path)) {
    if (fs::is_regular_file(entry)) {
      std::ifstream file(entry.path());
      if (file) {
        std::string text;
        {
          std::stringstream buffer;
          buffer << file.rdbuf();
          text = buffer.str();
        }

        std::sregex_token_iterator token(text.begin(), text.end(), re, -1);

        Review words;
        for (; token != end; ++token) {
          if (token->length() > 1) {  // don't use one letter words
            words.push_back(*token);
          }
        }
        max_size_ = std::max(max_size_, words.size());
        reviews.push_back(std::move(words));
      }
    }
  }
}

void GetWordsFrequencies(const ImdbReader& reader,
                         WordsFrequencies& frequencies) {
  for (size_t i = 0; i < reader.GetPosSize(); ++i) {
    const ImdbReader::Review& review = reader.GetPos(i);
    for (auto& word : review) {
      frequencies[word] += 1;
    }
  }

  for (size_t i = 0; i < reader.GetNegSize(); ++i) {
    const ImdbReader::Review& review = reader.GetNeg(i);
    for (auto& word : review) {
      frequencies[word] += 1;
    }
  }
}

void SelectTopFrequencies(WordsFrequencies& vocab, int64_t new_size) {
  using FreqItem = std::pair<size_t, WordsFrequencies::iterator>;
  std::vector<FreqItem> freq_items;
  freq_items.reserve(vocab.size());
  auto i = vocab.begin();
  auto e = vocab.end();
  for (; i != e; ++i) {
    freq_items.push_back({i->second, i});
  }

  std::sort(
      freq_items.begin(), freq_items.end(),
      [](const FreqItem& a, const FreqItem& b) { return a.first < b.first; });

  std::reverse(freq_items.begin(), freq_items.end());

  freq_items.resize(static_cast<size_t>(new_size));

  WordsFrequencies new_vocab;

  for (auto& item : freq_items) {
    new_vocab.insert({item.second->first, item.first});
  }

  vocab = new_vocab;
}
