#ifndef PAPER_H
#define PAPER_H

#include "review.h"

#include <vector>

struct Paper {
  uint32_t id{0};
  std::string preliminary_decision;
  std::vector<Review> reviews;
};

using Papers = std::vector<Paper>;

#endif  // PAPER_H
