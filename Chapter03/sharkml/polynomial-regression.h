#ifndef POLYNOMIALREGRESSION_H
#define POLYNOMIALREGRESSION_H

#include "monitor.h"
#include "polynomial-model.h"

#include <shark/Algorithms/GradientDescent/CG.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/AbsoluteLoss.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/Regularizer.h>

class PolynomialRegression : public shark::AbstractTrainer<PolynomialModel<>>,
                             public shark::IParameterizable<> {
 public:
  using MonitorType =
      Monitor<PolynomialModel<>,
              shark::LabeledData<shark::RealVector, shark::RealVector>,
              shark::AbsoluteLoss<>>;

  PolynomialRegression(double regularization = 0.0,
                       double polynomial_degree = 1,
                       int num_epochs = 100)
      : m_regularization{regularization},
        m_polynomial_degree{polynomial_degree},
        m_num_epochs{num_epochs} {}

  std::string name() const override { return "PolynomialRegression"; }

  double regularization() const { return m_regularization; }
  void setRegularization(double regularization) {
    RANGE_CHECK(regularization >= 0.0);
    m_regularization = regularization;
  }

  double polynomialDegree() const { return m_polynomial_degree; }
  void setPolynomialDegree(double polynomial_degree) {
    RANGE_CHECK(polynomial_degree >= 1.0);
    m_polynomial_degree = polynomial_degree;
  }

  shark::RealVector parameterVector() const override {
    shark::RealVector param(2);
    param(0) = m_regularization;
    param(0) = m_polynomial_degree;
    return param;
  }
  void setParameterVector(const shark::RealVector& param) override {
    SIZE_CHECK(param.size() == 2);
    m_regularization = param(0);
    m_polynomial_degree = param(1);
  }
  size_t numberOfParameters() const override { return 2; }

  void setMonitor(MonitorType* monitor) { m_monitor = monitor; }

  void train(PolynomialModel<>& model,
             shark::LabeledData<shark::RealVector, shark::RealVector> const&
                 dataset) override {
    std::size_t inputDim = inputDimension(dataset);
    std::size_t outputDim = labelDimension(dataset);

    model.setStructure(static_cast<size_t>(m_polynomial_degree), inputDim,
                       outputDim);

    shark::SquaredLoss<> loss;
    shark::SquaredLoss<> abs_loss;
    shark::ErrorFunction<> errorFunction(dataset, &model, &loss);
    // shark::TwoNormRegularizer<> regularizer;
    shark::OneNormRegularizer<> regularizer;
    errorFunction.setRegularizer(m_regularization, &regularizer);
    errorFunction.init();

    shark::CG<> optimizer;
    optimizer.init(errorFunction);
    if (m_monitor)
      m_monitor->clear();
    for (int i = 0; i != m_num_epochs; ++i) {
      optimizer.step(errorFunction);
      if (m_monitor) {
        auto pred = model(dataset.inputs());
        m_monitor->addTrainStep(abs_loss(dataset.labels(), pred));
        m_monitor->addValidationStep(model);
      }
    }
    // copy solution parameters into model
    model.setParameterVector(optimizer.solution().point);
  }

 protected:
  double m_regularization;
  double m_polynomial_degree;
  int m_num_epochs;
  MonitorType* m_monitor{nullptr};
};

#endif  // POLYNOMIALREGRESSION_H
