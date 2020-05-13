#ifndef DATA_H
#define DATA_H

#include <vector>

using DataType = double;
using Values = std::vector<DataType>;

Values LinSpace(double s, double e, size_t n);

std::pair<Values, Values> GenerateData(double s,
                                       double e,
                                       size_t n,
                                       size_t seed,
                                       bool noise);

#endif  // DATA_H
