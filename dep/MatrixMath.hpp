#ifndef MATRIXMATH_H
#define MATRIXMATH_H

#include <vector>

class MatrixMath{
  private:
    MatrixMath() {}

    static void swap(std::vector<double>& vecA, std::vector<double>& vecB);
  public:
    static double matrixMult1D(std::vector<double>& matrixA, std::vector<double>& matrixB);
    static std::vector<std::vector<double>> matrixMult2D(std::vector<std::vector<double>>& matrixA, std::vector<std::vector<double>>& matrixB);
    static std::vector<std::vector<double>> add(std::vector<std::vector<double>>& matrixA, std::vector<std::vector<double>>& matrixB);
    static std::vector<std::vector<double>> transpose2D(std::vector<std::vector<double>>& matrix);
    static std::vector<std::vector<double>> inverse(std::vector<std::vector<double>>& matrix);
    static double error(std::vector<std::vector<double>>& data, std::vector<double>& weight, std::vector<double>& output);
    static std::vector<double> regression(std::vector<std::vector<double>>& data, std::vector<double>& output);
};

#endif // MATRIXMATH_H
