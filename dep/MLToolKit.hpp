#ifndef MLTOOLKIT_H
#define MLTOOLKIT_H

#include "MatrixMath.hpp"

#include <iostream>
#include <string>
#include <vector>

class MLToolKit{
  private:
    std::vector<std::vector<double>> dataX;
    std::vector<double> output;
    std::vector<double> weight;

  public:
    MLToolKit(std::vector<double>& y, std::vector<std::vector<double>>& data);

    void Regression();
    std::vector<double> PerceptronLearning();
    int test(std::vector<double>& y, std::vector<std::vector<double>>& data, std::vector<double>& w);
    int test(std::vector<double>& y, std::vector<std::vector<double>>& data);
    std::vector<double> PocketLearning();
};

#endif //MLTOOLKIT_H

