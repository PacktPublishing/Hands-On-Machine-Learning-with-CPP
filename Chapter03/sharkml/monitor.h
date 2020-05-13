#ifndef MONITOR_H
#define MONITOR_H

#include <vector>

template <typename Model, typename Dataset, typename Loss>
class Monitor {
 public:
  void clear() {
    train_steps.clear();
    validation_steps.clear();
  }
  void addTrainStep(double value) { train_steps.push_back(value); }
  void addValidationStep(Model& model) {
    auto prediction = model(validation_data.inputs());
    validation_steps.push_back(loss(validation_data.labels(), prediction));
  }

  std::vector<double> train_steps;
  std::vector<double> validation_steps;
  Dataset validation_data;
  Loss loss;
};

#endif  // MONITOR_H
