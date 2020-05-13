#ifndef REVIEW_H
#define REVIEW_H

#include <string>

struct Review {
  std::string confidence;
  std::string evaluation;
  uint32_t id{0};
  std::string language;
  std::string orientation;
  std::string remarks;
  std::string text;
  std::string timespan;
};

#endif  // REVIEW_H
