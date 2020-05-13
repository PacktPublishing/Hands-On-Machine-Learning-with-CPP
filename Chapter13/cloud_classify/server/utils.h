#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>

using Classes = std::map<size_t, std::string>;

Classes ReadClasses(const std::string& file_name);

torch::Tensor ReadMemoryImageTensor(const std::string& data,
                                    int width,
                                    int height);

#endif  // UTILS_H
