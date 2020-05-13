#include <dlib/matrix.h>
#include <iostream>

int main() {
  // definitions
  {
    // compile time sized matrix
    dlib::matrix<double, 3, 1> y;
    // dynamically sized matrix
    dlib::matrix<double> m(3, 3);
    // later we can change size of this matrix
    m.set_size(6, 6);
  }
  // initializations
  {
    // comma operator
    dlib::matrix<double> m(3, 3);
    m = 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    std::cout << "Matix from comma operator\n" << m << std::endl;

    // wrap array
    double data[] = {1, 2, 3, 4, 5, 6};
    auto m2 = dlib::mat(data, 2, 3);  // create matrix with size 2x3
    std::cout << "Matix from array\n" << m2 << std::endl;

    // Matrix elements can be accessed with () operator
    m(1, 2) = 300;
    std::cout << "Matix element updated\n" << m << std::endl;

    // Also you can initialize matrix with some predefined values
    auto a = dlib::identity_matrix<double>(3);
    std::cout << "Identity matix \n" << a << std::endl;

    auto b = dlib::ones_matrix<double>(3, 4);
    std::cout << "Ones matix \n" << b << std::endl;

    auto c = dlib::randm(3, 4);  // matrix with random values with size 3x3
    std::cout << "Random matix \n" << c << std::endl;
  }
  // arithmetic operations
  {
    dlib::matrix<double> a(2, 2);
    a = 1, 1, 1, 1;
    dlib::matrix<double> b(2, 2);
    b = 2, 2, 2, 2;

    auto c = a + b;
    std::cout << "c = a + b \n" << c << std::endl;

    auto e = a * b;  // real matrix multiplication
    std::cout << "e = a dot b \n" << e << std::endl;

    a += 5;
    std::cout << "a += 5 \n" << a << std::endl;

    auto d = dlib::pointwise_multiply(a, b);  // element wise multiplication
    std::cout << "d = a * b \n" << e << std::endl;

    auto t = dlib::trans(a);  // transpose matrix
    std::cout << "transposed matrix a \n" << t << std::endl;
  }
  // partial access
  {
    dlib::matrix<float, 4, 4> m;
    m = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    auto sm =
        dlib::subm(m, dlib::range(1, 2),
                   dlib::range(1, 2));  // original matrix can't be updated
    std::cout << "Sub matrix \n" << sm << std::endl;

    dlib::set_subm(m, dlib::range(1, 2), dlib::range(1, 2)) = 100;
    std::cout << "Updated sub matrix \n" << m << std::endl;
  }
  // there are no implicit broadcasting in dlib
  {
    // we can simulate broadcasting with partial access
    dlib::matrix<float, 2, 1> v;
    v = 10, 10;
    dlib::matrix<float, 2, 3> m;
    m = 1, 2, 3, 4, 5, 6;
    for (int i = 0; i < m.nc(); ++i) {
      dlib::set_colm(m, i) += v;
    }
    std::cout << "Matrix with updated columns \n" << m << std::endl;
  }
  return 0;
}
