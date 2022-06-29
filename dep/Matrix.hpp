#ifndef MATRIXMATH_H
#define MATRIXMATH_H

#include <string>
#include <vector>

class Matrix{
  private:
    std::vector<std::vector<double>> matrix;

    double matrixMult1D(std::vector<double>& matrixA, std::vector<double>& matrixB);
    void swap(std::vector<double>& vecA, std::vector<double>& vecB);

  public:
    Matrix(std::vector<std::vector<double>>& matrix);
    Matrix(std::vector<double>& matrix);
    Matrix(){};
    int size(){return matrix.size();};
    std::vector<double>& at(int i);
    Matrix operator*(Matrix& matrixB);
    Matrix operator*(double coefficient);
    Matrix operator+(Matrix& matrixB);
    Matrix operator-(Matrix matrixB);
    Matrix transpose();
    Matrix inverse();
    double squaredMagnitude();
    std::string toString();
};

#endif // MATRIXMATH_H
