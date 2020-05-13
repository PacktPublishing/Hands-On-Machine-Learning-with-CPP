#include "plot.h"

#include <plot.h>

#include <algorithm>

using namespace shark;

namespace {
template <typename T>
auto DataToVector(const T& data) {
  std::vector<double> vec;
  vec.reserve(data.shape()[0]);
  std::for_each(data.elements().begin(), data.elements().end(),
                [&](auto elem) { vec.emplace_back(elem[0]); });
  return vec;
}
}  // namespace

void PlotData(const Data<RealVector>& x,
              const Data<RealVector>& y,
              const Data<RealVector>& x_val,
              const Data<RealVector>& y_val,
              const Data<RealVector>& x_pred,
              const Data<RealVector>& y_pred) {
  plotcpp::Plot plt;
  plt.SetTerminal("png");
  plt.SetOutput("plot.png");
  plt.SetTitle("Polynomial regression");
  plt.SetXLabel("x");
  plt.SetYLabel("y");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto x_coords = DataToVector(x);
  auto y_coords = DataToVector(y);

  auto x_val_coords = DataToVector(x_val);
  auto y_val_coords = DataToVector(y_val);

  auto x_pred_coords = DataToVector(x_pred);
  auto y_pred_coords = DataToVector(y_pred);

  plt.Draw2D(
      plotcpp::Points(x_coords.begin(), x_coords.end(), y_coords.begin(),
                      "orig", "lc rgb 'black' pt 7"),
      plotcpp::Points(x_val_coords.begin(), x_val_coords.end(),
                      y_val_coords.begin(), "val", "lc rgb 'green' pt 7"),
      plotcpp::Lines(x_pred_coords.begin(), x_pred_coords.end(),
                     y_pred_coords.begin(), "pred", "lc rgb 'red' lw 2"));
  plt.Flush();
}

void PlotTrain(const std::vector<double>& train,
               const std::vector<double>& validation) {
  plotcpp::Plot plt;
  plt.SetTerminal("png");
  plt.SetOutput("train_plot.png");
  plt.SetTitle("Losses");
  plt.SetXLabel("num iteration");
  plt.SetYLabel("loss");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  std::vector<double> x_coords(train.size());
  std::iota(x_coords.begin(), x_coords.end(), 0);

  plt.Draw2D(
      plotcpp::Lines(x_coords.begin(), x_coords.end(), train.begin(), "train",
                     "lc rgb 'black' lw 2"),
      plotcpp::Lines(x_coords.begin(), x_coords.end(), validation.begin(),
                     "validation", "lc rgb 'red' lw 2"));
  plt.Flush();
}
