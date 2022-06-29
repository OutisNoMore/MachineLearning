#ifndef MLTOOLKIT_H
#define MLTOOLKIT_H

#include "Matrix.hpp"

#include <vector>
#include <cmath>

class MLToolKit{
  private:
    Matrix dataX;
    Matrix output;
    Matrix weight;

    double LogisticFunction(double signal);
    Matrix GradientError();

  public:
    MLToolKit(std::vector<double>& y, std::vector<std::vector<double>>& data);

    double error();
    Matrix w(){return weight;}
    void LinearRegression();
    Matrix PerceptronLearning();
    Matrix PocketLearning();
    Matrix LogisticRegression();
    int test(std::vector<double>& y, std::vector<std::vector<double>>& data, Matrix w);
    int test(std::vector<double>& y, std::vector<std::vector<double>>& data);
    int testLogistic(std::vector<double>& y, std::vector<std::vector<double>>& data);
};

#endif //MLTOOLKIT_H

