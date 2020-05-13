#include <shark/LinAlg/BLAS/remora.hpp>

#include <iostream>
#include <vector>

template <class VecV>
void vec_sum(
    remora::vector_expression<VecV, typename VecV::device_type> const& v) {
  auto const& elem_result = eval_block(v);
}
int main() {
  // definitions
  {
    // dynamically sized array dense vector.
    remora::vector<double> b(100,
                             1.0);  // vector of size 100 and filled with 1.0

    // dynamically sized array dense matrix
    remora::matrix<double> C(2, 2);  // 2x2 matrix
  }
  // initializations
  {
    // fill with one value
    remora::matrix<float> m_zero(2, 2, 0.0f);
    std::cout << "Zero matrix \n" << m_zero << std::endl;

    // initializer list
    remora::matrix<float> m_ones{{1, 1}, {1, 1}};
    std::cout << "Initializer list matrix \n" << m_ones << std::endl;

    // wrap c++ array
    std::vector<float> data{1, 2, 3, 4};
    auto m = remora::dense_matrix_adaptor<float>(data.data(), 2, 2);
    std::cout << "Wrapped array matrix \n" << m << std::endl;
    auto v = remora::dense_vector_adaptor<float>(data.data(), 4);
    std::cout << "Wrapped array vector \n" << v << std::endl;

    // Direct access to container elements
    m(0, 0) = 3.14f;
    std::cout << "Changed matrix element\n" << m << std::endl;
    v(0) = 3.14f;
    std::cout << "Changed vector element\n" << v << std::endl;
  }
  // Arithmetic operations examples
  {
    remora::matrix<float> a{{1, 1}, {1, 1}};
    remora::matrix<float> b{{2, 2}, {2, 2}};

    remora::matrix<float> c = a + b;
    std::cout << "c = a + b\n" << c << std::endl;

    a -= b;
    std::cout << "a -= b\n" << a << std::endl;

    c = remora::prod(a, b);
    std::cout << "c = a dot b\n" << c << std::endl;

    c = a % b;  // also dot product
    std::cout << "c = a % b\n" << c << std::endl;

    c = a * b;  // element wise product
    std::cout << "c = a *b\n" << c << std::endl;

    c = a + 5;
    std::cout << "c = a + 5\n" << c << std::endl;
  }
  // partial access
  {
    remora::matrix<float> m{
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto r = remora::rows(m, 0, 2);
    std::cout << "Matrix rows 0,2\n" << r << std::endl;

    auto sr = remora::subrange(m, 1, 3, 1, 3);
    std::cout << "Matrix subrange\n" << sr << std::endl;
    sr *= 67;
    std::cout << "Matrix with updated subrange\n" << m << std::endl;
  }
  // broadcasting is not supported directly
  {
    // Reductions
    remora::matrix<float> m{{1, 2, 3, 4}, {5, 6, 7, 8}};
    remora::vector<float> v{10, 10};
    auto cols = remora::as_columns(m);
    std::cout << "Sum reduction for columns\n"
              << remora::sum(cols) << std::endl;

    // Update matrix rows
    for (size_t i = 0; i < m.size2(); ++i) {
      remora::column(m, i) += v;
    }
    std::cout << "Updated rows\n" << m << std::endl;
  }
  return 0;
}
