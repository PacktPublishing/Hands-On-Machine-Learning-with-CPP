#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <iostream>
#include <vector>

int main() {
  {
    // declaration of dynamically sized array
    {
      std::vector<uint64_t> shape = {3, 2, 4};
      xt::xarray<double, xt::layout_type::row_major> a(shape);
    }
    // declaration of dynamically sized tensor with fixed dimmentions number
    {
      std::array<size_t, 3> shape = {3, 2, 4};
      xt::xtensor<double, 3> a(shape);
    }

    // declaration of tensor with shape fixed at compile time.
    { xt::xtensor_fixed<double, xt::xshape<3, 2, 4>> a; }

    // Initialization of xtensor arrays can be done with C++ initializer lists:
    {
      xt::xarray<double> arr1{{1.0, 2.0, 3.0},
                              {2.0, 5.0, 7.0},
                              {2.0, 5.0, 7.0}};  // initialize a 3x3 array
      std::cout << "Tensor from initializer list :\n" << arr1 << std::endl;
    }
    // Special types of initializers
    {
      std::vector<uint64_t> shape = {2, 2};
      std::cout << "Ones matrix :\n" << xt::ones<float>(shape) << std::endl;
      std::cout << "Zero matrix :\n" << xt::zeros<float>(shape) << std::endl;
      std::cout << "Matrix with ones on the diagonal:\n"
                << xt::eye<float>(shape) << std::endl;
    }
    // Mapping c++ array to tensors
    {
      std::vector<float> data{1, 2, 3, 4};
      std::vector<size_t> shape{2, 2};
      auto data_x = xt::adapt(data, shape);
      std::cout << "Matrix from vector :\n" << data_x << std::endl;
    }
    // Element access
    {
      std::vector<size_t> shape = {3, 2, 4};
      xt::xarray<float> a = xt::ones<float>(shape);
      a(2, 1, 3) = 3.14f;
      std::cout << "Updated element :\n" << a << std::endl;
    }
    // Arithmetic operations examples
    {
      xt::xarray<double> a = xt::random::rand<double>({2, 2});
      xt::xarray<double> b = xt::random::rand<double>({2, 2});

      std::cout << "A: \n" << a << std::endl;
      std::cout << "B: \n" << b << std::endl;

      xt::xarray<double> c = a + b;
      std::cout << "c = a + b \n" << c << std::endl;
      a -= b;
      std::cout << "a -= b \n" << a << std::endl;
      c = xt::linalg::dot(a, b);
      std::cout << "a dot b \n" << c << std::endl;
      c = a + 5;
      std::cout << "c = a + 5 \n" << c << std::endl;
      c = a * b;
      std::cout << "c = a * b \n" << c << std::endl;
    }
    // Partial access to xtensor containers
    {
      xt::xarray<int> a{
          {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
      auto b = xt::view(a, xt::range(1, 3), xt::range(1, 3));
      std::cout << "Partial view on a \n" << b << std::endl;
    }
    // Broadcasting
    {
      auto a = xt::xarray<double>({{1, 2}, {3, 4}});
      auto b = xt::xarray<double>({10, 20});
      b.reshape({2, 1});
      std::cout << "A: \n" << a << std::endl;
      std::cout << "B: \n" << b << std::endl;
      auto c = a + b;
      std::cout << "Columns broadcasting: \n" << c << std::endl;
    }
  }
  return 0;
};
