#include "reviewsreader.h"

#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>

#include <memory>

enum class HandlerState {
  None,
  Global,
  PapersArray,
  Paper,
  ReviewArray,
  Review
};

struct ReviewsHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, ReviewsHandler> {
  ReviewsHandler(Papers* papers) : papers_(papers) {}

  bool Uint(unsigned u) {
    bool res{true};
    try {
      if (state_ == HandlerState::Paper && key_ == "id") {
        paper_.id = u;
      } else if (state_ == HandlerState::Review && key_ == "id") {
        review_.id = u;
      } else {
        res = false;
      }
    } catch (...) {
      res = false;
    }
    key_.clear();
    return res;
  }

  bool String(const char* str, rapidjson::SizeType length, bool /*copy*/) {
    bool res{true};
    try {
      if (state_ == HandlerState::Paper && key_ == "preliminary_decision") {
        paper_.preliminary_decision = std::string(str, length);
      } else if (state_ == HandlerState::Review && key_ == "confidence") {
        review_.confidence = std::string(str, length);
      } else if (state_ == HandlerState::Review && key_ == "evaluation") {
        review_.evaluation = std::string(str, length);
      } else if (state_ == HandlerState::Review && key_ == "lan") {
        review_.language = std::string(str, length);
      } else if (state_ == HandlerState::Review && key_ == "orientation") {
        review_.orientation = std::string(str, length);
      } else if (state_ == HandlerState::Review && key_ == "remarks") {
        review_.remarks = std::string(str, length);
      } else if (state_ == HandlerState::Review && key_ == "text") {
        review_.text = std::string(str, length);
      } else if (state_ == HandlerState::Review && key_ == "timespan") {
        review_.timespan = std::string(str, length);
      } else {
        res = false;
      }
    } catch (...) {
      res = false;
    }
    key_.clear();
    return res;
  }

  bool Key(const char* str, rapidjson::SizeType length, bool /*copy*/) {
    key_ = std::string(str, length);
    return true;
  }

  bool StartObject() {
    if (state_ == HandlerState::None && key_.empty()) {
      state_ = HandlerState::Global;
    } else if (state_ == HandlerState::PapersArray && key_.empty()) {
      state_ = HandlerState::Paper;
    } else if (state_ == HandlerState::ReviewArray && key_.empty()) {
      state_ = HandlerState::Review;
    } else {
      return false;
    }
    return true;
  }

  bool EndObject(rapidjson::SizeType /*memberCount*/) {
    if (state_ == HandlerState::Global) {
      state_ = HandlerState::None;
    } else if (state_ == HandlerState::Paper) {
      state_ = HandlerState::PapersArray;
      papers_->push_back(paper_);
      paper_ = Paper();
    } else if (state_ == HandlerState::Review) {
      state_ = HandlerState::ReviewArray;
      paper_.reviews.push_back(review_);
    } else {
      return false;
    }
    return true;
  }

  bool StartArray() {
    if (state_ == HandlerState::Global && key_ == "paper") {
      state_ = HandlerState::PapersArray;
      key_.clear();
    } else if (state_ == HandlerState::Paper && key_ == "review") {
      state_ = HandlerState::ReviewArray;
      key_.clear();
    } else {
      return false;
    }
    return true;
  }

  bool EndArray(rapidjson::SizeType /*elementCount*/) {
    if (state_ == HandlerState::ReviewArray) {
      state_ = HandlerState::Paper;
    } else if (state_ == HandlerState::PapersArray) {
      state_ = HandlerState::Global;
    } else {
      return false;
    }
    return true;
  }

  Paper paper_;
  Review review_;
  std::string key_;
  Papers* papers_{nullptr};
  HandlerState state_{HandlerState::None};
};

Papers ReadPapersReviews(const std::string& filename) {
  auto file = std::unique_ptr<FILE, void (*)(FILE*)>(
      fopen(filename.c_str(), "r"), [](FILE* f) {
        if (f)
          ::fclose(f);
      });
  if (file) {
    char readBuffer[65536];
    rapidjson::FileReadStream is(file.get(), readBuffer, sizeof(readBuffer));
    rapidjson::Reader reader;
    Papers papers;
    ReviewsHandler handler(&papers);
    auto res = reader.Parse(is, handler);
    if (!res) {
      throw std::runtime_error(rapidjson::GetParseError_En(res.Code()));
    }
    return papers;
  } else {
    throw std::invalid_argument("File can't be opened " + filename);
  }
}
