#include "data.h"

using namespace shark;

Data<RealVector> LinSpace(double s, double e, size_t n) {
  std::vector<RealVector> x_data(n);
  // generate data
  RealVector x_v(1);  // it's a typdef to remora::vector<float>

  double step = (e - s) / n;

  double v = s;
  for (size_t i = 0; i < n; ++i) {
    x_v(0) = v;
    x_data[i] = x_v;
    v += step;
  }
  x_v(0) = e;
  x_data[n - 1] = x_v;

  return createDataFromRange(x_data);
}

namespace {

const std::vector<double> x_gen{
    0.0202184,  0.07103606, 0.0871293,  0.11827443, 0.14335329, 0.38344152,
    0.41466194, 0.4236548,  0.43758721, 0.46147936, 0.52184832, 0.52889492,
    0.54488318, 0.5488135,  0.56804456, 0.60276338, 0.63992102, 0.64589411,
    0.71518937, 0.77815675, 0.78052918, 0.79172504, 0.79915856, 0.83261985,
    0.87001215, 0.891773,   0.92559664, 0.94466892, 0.96366276, 0.97861834};

const std::vector<double> y_gen{
    1.0819082,   0.87027612,  1.14386208,  0.70322051,  0.78494746,
    -0.25265944, -0.22066063, -0.26595867, -0.4562644,  -0.53001927,
    -0.86481449, -0.99462675, -0.87458603, -0.83407054, -0.77090649,
    -0.83476183, -1.03080067, -1.02544303, -1.0788268,  -1.00713288,
    -1.03009698, -0.63623922, -0.86230652, -0.75328767, -0.70023795,
    -0.41043495, -0.50486767, -0.27907117, -0.25994628, -0.06189804};

std::pair<Data<RealVector>, Data<RealVector>> GetData(
    const std::vector<double>& xx,
    const std::vector<double>& yy) {
  auto n = xx.size();
  std::vector<RealVector> x_data(n);
  std::vector<RealVector> y_data(n);

  RealVector x_v(1);
  RealVector y_v(1);

  size_t i = 0;
  for (auto x : xx) {
    x_v(0) = x;
    x_data[i] = x_v;
    ++i;
  }
  i = 0;
  for (auto y : yy) {
    y_v(0) = y;
    y_data[i] = y_v;
    ++i;
  }

  return {createDataFromRange(x_data), createDataFromRange(y_data)};
}

const std::vector<double> x_val_gen = {
    0.00936602, 0.0626953,  0.07702452, 0.11339586, 0.13351575, 0.21442521,
    0.22320716, 0.23853779, 0.27506891, 0.3087274,  0.43859577, 0.45918468,
    0.46066071, 0.46296798, 0.53803913, 0.60767948, 0.64317051, 0.67384069,
    0.7811449,  0.79062151, 0.81326169, 0.82488721, 0.87496673, 0.87819742,
    0.92493664, 0.92900401, 0.94038862, 0.95022127, 0.97375838, 0.97749171};

const std::vector<double> y_val_gen = {
    0.99902615,  0.95667277,  0.93484662,  0.86059201,  0.80851142,
    0.53147536,  0.49597504,  0.43200384,  0.27112791,  0.11569305,
    -0.4759444,  -0.55890152, -0.56465577, -0.5735958,  -0.82185147,
    -0.9616143,  -0.99387647, -0.99942861, -0.85798313, -0.8341958,
    -0.77072839, -0.73468319, -0.5557006,  -0.54297949, -0.34639715,
    -0.32835428, -0.27723205, -0.23243133, -0.12334578, -0.10586903};
}  // namespace

std::pair<shark::Data<shark::RealVector>, shark::Data<shark::RealVector>>
GetXYData() {
  return GetData(x_gen, y_gen);
}

std::pair<shark::Data<shark::RealVector>, shark::Data<shark::RealVector>>
GetXYValidationData() {
  return GetData(x_val_gen, y_val_gen);
}

std::pair<Data<RealVector>, Data<RealVector>> GenerateData(size_t num_samples,
                                                           bool no_noise,
                                                           size_t seed) {
  std::vector<RealVector> x_data(num_samples);
  std::vector<RealVector> y_data(num_samples);

  std::mt19937 re(seed);
  std::normal_distribution<double> dist;

  // generate data
  RealVector x_v(1);
  RealVector y_v(1);

  for (size_t i = 0; i < num_samples;) {
    x_v(0) = dist(re);
    x_data[i] = x_v;

    y_v(0) = std::cos(M_PI * x_v(0)) + (no_noise ? 0 : (dist(re) * 0.3));
    y_data[i] = y_v;
    ++i;
  }

  return {createDataFromRange(x_data), createDataFromRange(y_data)};
}
