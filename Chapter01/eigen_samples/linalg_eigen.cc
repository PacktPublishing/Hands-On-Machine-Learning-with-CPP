#include <Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<float, 3, 3> MyMatrix33f;
typedef Eigen::Matrix<float, 3, 1> MyVector3f;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MyMatrix;

int main() {
  {
    // declaration
    MyMatrix33f a;
    MyVector3f v;
    MyMatrix m(10, 15);

    // initialization
    a = MyMatrix33f::Zero();
    std::cout << "Zero matrix:\n" << a << std::endl;

    a = MyMatrix33f::Identity();
    std::cout << "Identity matrix:\n" << a << std::endl;

    v = MyVector3f::Random();
    std::cout << "Random vector:\n" << v << std::endl;

    a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    std::cout << "Comma initilized matrix:\n" << a << std::endl;

    a(0, 0) = 3;
    std::cout << "Matrix with changed element[0][0]:\n" << a << std::endl;

    int data[] = {1, 2, 3, 4};
    Eigen::Map<Eigen::RowVectorXi> v_map(data, 4);
    std::cout << "Row vector mapped to array:\n" << v_map << std::endl;

    std::vector<float> vdata = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Eigen::Map<MyMatrix33f> a_map(vdata.data());
    std::cout << "Matrix mapped to array:\n" << a_map << std::endl;
  }
  // arithmetic
  {
    Eigen::Matrix2d a;
    a << 1, 2, 3, 4;
    Eigen::Matrix2d b;
    b << 1, 2, 3, 4;

    // element wise operations
    Eigen::Matrix2d result = a.array() * b.array();
    std::cout << "element wise a * b :\n" << result << std::endl;

    result = a.array() / b.array();
    std::cout << "element wise a / b :\n" << result << std::endl;

    a = b.array() * 4;
    std::cout << "element wise a = b * 4 :\n" << a << std::endl;

    // matrix operations
    result = a + b;
    std::cout << "matrices a + b :\n" << result << std::endl;

    a += b;
    std::cout << "matrices a += b :\n" << result << std::endl;

    result = a * b;
    std::cout << "matrices a * b :\n" << result << std::endl;
  }

  // patial access
  {
    Eigen::MatrixXf m = Eigen::MatrixXf::Random(4, 4);
    std::cout << "Random 4x4 matrix :\n" << m << std::endl;

    Eigen::Matrix2f b =
        m.block(1, 1, 2, 2);  // coping the middle part of matrix
    std::cout << "Middle of 4x4 matrix :\n" << b << std::endl;

    m.block(1, 1, 2, 2) *= 0;  // change values in original matrix
    std::cout << "Modified middle of 4x4 matrix :\n" << m << std::endl;

    m.row(1).array() += 3;
    std::cout << "Modified row of 4x4 matrix :\n" << m << std::endl;

    m.col(2).array() /= 4;
    std::cout << "Modified col of 4x4 matrix :\n" << m << std::endl;
  }

  // broadcasting
  {
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(2, 4);
    std::cout << "Random 2x4 matrix :\n" << mat << std::endl;

    Eigen::VectorXf v(2);  // column vector
    v << 100, 100;
    mat.colwise() += v;
    std::cout << "Sum broadcasted over columns :\n" << mat << std::endl;
  }
  return 0;
};
