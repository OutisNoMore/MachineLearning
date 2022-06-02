#ifndef MLTOOLKIT_H
#define MLTOOLKIT_H

#include "MatrixMath.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

class MLToolKit{
  private:
    std::vector<std::vector<double>> dataX;
    std::vector<double> output;
    std::vector<double> weight;

  public:
    MLToolKit(std::string fileName);

    void Regression();
    void PerceptronLearning();
    void test(std::string fileName, std::vector<double>& w);
    void test(std::string fileName);
    std::vector<double> PocketLearning();
};

#endif //MLTOOLKIT_H

