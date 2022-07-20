#ifndef MLTOOLKIT_H
#define MLTOOLKIT_H

#include "Matrix.hpp"

#include <vector>
#include <cmath>
#include <cstdlib>

class MLToolKit{
  private:
    Matrix dataX;
    Matrix output;
    Matrix weight;

    double LogisticFunction(double signal);
    Matrix GradientError();
    double ThetaFunction(double signal);

  public:
    MLToolKit(std::vector<double>& y, std::vector<std::vector<double>>& data);

    double error();
    Matrix w(){return weight;}
    void LinearRegression();
    Matrix PerceptronLearning();
    Matrix PocketLearning();
    Matrix LogisticRegression();
    std::vector<std::vector<std::vector<double>>> NeuralNetwork();
    int test(std::vector<double>& y, std::vector<std::vector<double>>& data, Matrix w);
    int test(std::vector<double>& y, std::vector<std::vector<double>>& data);
    int testLogistic(std::vector<double>& y, std::vector<std::vector<double>>& data);
    int testNeural(std::vector<double>& y, std::vector<std::vector<double>>& data, std::vector<std::vector<std::vector<double>>>& weights);
};

#endif //MLTOOLKIT_H

