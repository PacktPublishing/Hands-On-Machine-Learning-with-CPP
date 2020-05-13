#ifndef DATA_H
#define DATA_H

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

shogun::SGMatrix<float64_t> LinSpace(double s, double e, size_t n);

std::pair<shogun::SGMatrix<float64_t>, shogun::SGVector<float64_t>>
GenerateData(size_t num_samples, size_t seed, bool no_noise);

#endif  // DATA_H
