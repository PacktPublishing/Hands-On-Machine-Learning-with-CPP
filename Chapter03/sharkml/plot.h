#ifndef PLOT_H
#define PLOT_H

#include <shark/Data/Dataset.h>

#include <vector>

void PlotData(const shark::Data<shark::RealVector>& x,
              const shark::Data<shark::RealVector>& y,
              const shark::Data<shark::RealVector>& x_val,
              const shark::Data<shark::RealVector>& y_val,
              const shark::Data<shark::RealVector>& x_pred,
              const shark::Data<shark::RealVector>& y_pred);

void PlotTrain(const std::vector<double>& train,
               const std::vector<double>& validation);

#endif  // PLOT_H
