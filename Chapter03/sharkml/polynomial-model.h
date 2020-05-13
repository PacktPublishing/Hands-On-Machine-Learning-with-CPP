#ifndef POLYNOMIALMODEL_H
#define POLYNOMIALMODEL_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/LinearModel.h>
#include <shark/Models/NeuronLayers.h>

template <class InputType = shark::RealVector,
          class ActivationFunction = shark::LinearNeuron>
class PolynomialModel
    : public shark::AbstractModel<
          InputType,
          shark::blas::vector<
              typename InputType::value_type,
              typename InputType::device_type>,  // type of output uses
                                                 // same device and
                                                 // precision as input
          shark::blas::vector<
              typename InputType::value_type,
              typename InputType::device_type>  // type of parameters
                                                // uses same device and
                                                // precision as input
          > {
 public:
  typedef shark::blas::vector<typename InputType::value_type,
                              typename InputType::device_type>
      VectorType;
  typedef shark::blas::matrix<typename InputType::value_type,
                              shark::blas::row_major,
                              typename InputType::device_type>
      MatrixType;

  shark::LinearModel<InputType, ActivationFunction> m_model;

 private:
  typedef shark::AbstractModel<InputType, VectorType, VectorType> base_type;
  typedef PolynomialModel<InputType, ActivationFunction> self_type;

  shark::Shape m_inputShape;
  size_t m_polynomial_degree{1};

 public:
  typedef typename base_type::BatchInputType BatchInputType;
  typedef typename base_type::BatchOutputType
      BatchOutputType;  // same as MatrixType
  typedef typename base_type::ParameterVectorType
      ParameterVectorType;  // same as VectorType

  /// CDefault Constructor; use setStructure later
  PolynomialModel() {
    this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
    if (std::is_base_of<shark::blas::dense_tag,
                        typename InputType::storage_type::storage_tag>::value) {
      this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
    }
  }

  /// \brief From INameable: return the class name.
  std::string name() const override { return "PolynomialModel"; }

  /// check for the presence of an offset term
  bool hasOffset() const { return m_model.hasOffset(); }

  ///\brief Returns the expected shape of the input
  shark::Shape inputShape() const override { return m_inputShape; }
  ///\brief Returns the shape of the output
  shark::Shape outputShape() const override { return m_model.outputShape(); }

  /// obtain the parameter vector
  ParameterVectorType parameterVector() const override {
    return m_model.parameterVector();
  }

  /// overwrite the parameter vector
  void setParameterVector(ParameterVectorType const& newParameters) override {
    m_model.setParameterVector(newParameters);
  }

  /// return the number of parameter
  size_t numberOfParameters() const override {
    return m_model.numberOfParameters();
  }

  /// overwrite structure and parameters
  void setStructure(size_t polynomial_degree,
                    shark::Shape const& inputs,
                    shark::Shape const& outputs = 1,
                    bool offset = false) {
    m_inputShape = inputs;

    m_polynomial_degree = polynomial_degree;
    size_t size = static_cast<size_t>(
        std::pow(inputs.numElements() + 1, polynomial_degree));

    shark::LinearModel<InputType, ActivationFunction> model(size, outputs,
                                                            offset);

    m_model = model;
  }

  /// return a copy of the matrix in dense format
  MatrixType const& matrix() const { return m_model.matrix(); }

  MatrixType& matrix() { return m_model.matrix(); }

  /// return the offset
  VectorType const& offset() const { return m_model.offset(); }
  VectorType& offset() { return m_model.offset(); }

  /// \brief Returns the activation function.
  ActivationFunction const& activationFunction() const {
    return m_model.activationFunction();
  }

  /// \brief Returns the activation function.
  ActivationFunction& activationFunction() {
    return m_model.activationFunction();
  }

  boost::shared_ptr<shark::State> createState() const override {
    return m_model.createState();
  }

  using base_type::eval;

  VectorType toPolynomial(VectorType const& vec) const {
    VectorType base = vec;
    base.push_back(1.0);
    VectorType out = base;
    for (size_t d = 2; d < m_polynomial_degree + 1; ++d) {
      auto size = out.size() * base.size();
      VectorType res(size);
      size_t res_i = 0;
      for (size_t i = 0; i < base.size(); ++i) {
        for (size_t j = 0; j < out.size(); ++j) {
          res[res_i] = base[i] * out[j];
          ++res_i;
        }
      }
      out = res;
    }

    return out;
  }

  BatchInputType modify_inputs(BatchInputType const& inputs) const {
    BatchInputType new_inputs(inputs.size1(),
                              m_model.inputShape().numElements());
    for (size_t row = 0; row < inputs.size1(); ++row) {
      for (size_t col = 0; col < inputs.size2(); ++col) {
        shark::blas::row(new_inputs, row) =
            toPolynomial(shark::blas::row(inputs, row));
      }
    }
    return new_inputs;
  }

  /// Evaluate the model: output = matrix * input + offset
  void eval(BatchInputType const& inputs,
            BatchOutputType& outputs,
            shark::State& state) const override {
    m_model.eval(modify_inputs(inputs), outputs, state);
  }

  ///\brief Calculates the first derivative w.r.t the parameters and summing
  /// them up over all patterns of the last computed batch
  void weightedParameterDerivative(
      BatchInputType const& patterns,
      BatchOutputType const& outputs,
      BatchOutputType const& coefficients,
      shark::State const& state,
      ParameterVectorType& gradient) const override {
    m_model.weightedParameterDerivative(modify_inputs(patterns), outputs,
                                        coefficients, state, gradient);
  }

  /// From ISerializable
  void read(shark::InArchive& archive) override {
    m_model.read(archive);
    archive >> m_inputShape;
  }
  /// From ISerializable
  void write(shark::OutArchive& archive) const override {
    m_model.write(archive);
    archive << m_inputShape;
  }
};

#endif  // POLYNOMIALMODEL_H
