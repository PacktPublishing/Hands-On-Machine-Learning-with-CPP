#ifndef DATA_H
#define DATA_H

#include <shark/Data/Dataset.h>

shark::Data<shark::RealVector> LinSpace(double s, double e, size_t n);

std::pair<shark::Data<shark::RealVector>, shark::Data<shark::RealVector>>
GenerateData(size_t num_samples, bool no_noise, size_t seed);

std::pair<shark::Data<shark::RealVector>, shark::Data<shark::RealVector>>
GetXYData();

std::pair<shark::Data<shark::RealVector>, shark::Data<shark::RealVector>>
GetXYValidationData();

template <typename D>
auto DataMin(const D& data) {
  auto elems = data.elements();
  auto mm = std::minmax_element(
      elems.begin(), elems.end(),
      [](const auto& a, const auto& b) { return a(0) < b(0); });
  return std::make_pair((*mm.first)(0), (*mm.second)(0));
}

#endif  // DATA_H
