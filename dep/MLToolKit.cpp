#include <iostream>
#include "MLToolKit.hpp"

MLToolKit::MLToolKit(std::vector<double>& y, std::vector<std::vector<double>>& data){
  output = Matrix{y};
  output = output.transpose();
  dataX = Matrix{data};
}

double MLToolKit::error(){
  Matrix out = dataX * weight;
  Matrix err = output - out;
  double error = err.squaredMagnitude();
  error = error / dataX.size();

  return error;
}

void MLToolKit::LinearRegression(){
  Matrix transposed = dataX.transpose();
  Matrix mult = transposed * dataX;
  Matrix inversed = mult.inverse();
  Matrix pseudoInverse = inversed * transposed;
  weight = pseudoInverse * output;
}

Matrix MLToolKit::PerceptronLearning(){
  // Get initial weights using Linear Regression
  LinearRegression();
  for(int i = 0; i < dataX.size(); i++){
    Matrix x(dataX.at(i));
    Matrix mult = x * weight;
    double hyp = mult.at(0)[0];
    if(hyp * (output.at(i))[0] < 0){
      // Mis-classified data
      mult = x * (output.at(i)[0]);
      mult = mult.transpose();
      weight = weight + mult;
    }
  }
  return weight;
}

Matrix MLToolKit::PocketLearning(){
  LinearRegression();
  double err = error();
  Matrix pocket = weight;
  for(int i = 0; i < dataX.size(); i++){
    Matrix x(dataX.at(i));
    Matrix mult = x * weight;
    double hyp = mult.at(0)[0];
    if(hyp * (output.at(i))[0] < 0){
      // Mis-classified data
      mult = x * (output.at(i)[0]);
      mult = mult.transpose();
      weight = weight + mult;
      double test = error();
      if(test < err){
        pocket = weight;
        err = test;
      }
    }
  }
  return pocket;
}


double MLToolKit::LogisticFunction(double signal){
  double theta = 1.0/(1 + exp(-1*signal));
  return theta;
}

Matrix MLToolKit::GradientError(){
  std::vector<double> error(dataX.at(0).size(), 0.0);
  Matrix stoError(error);
 
  for(int i = 0; i < dataX.size(); i++){
    Matrix x(dataX.at(i));
    Matrix power = x * weight;
    power = power * output.at(i)[0];
    double coefficient = power.at(0)[0];
    coefficient = 1 + exp(coefficient);
    coefficient = output.at(i)[0] / coefficient;
    x = x * coefficient;
    stoError = stoError + x;
  }

  stoError = stoError * (-1.0 / dataX.size());

  return stoError;
}

Matrix MLToolKit::LogisticRegression(){
  std::vector<double> w(dataX.at(0).size(), 0.0);
  weight = Matrix{w};
  weight = weight.transpose();
  double stepSize = 0.1; // step size of SGD
  // Maximum of 10,000 loops
  for(int i = 0; i < 10000; i++){
    Matrix error = GradientError(); // Get error
    double magnitude = error.squaredMagnitude();
    std::cout << magnitude << std::endl;
    magnitude = sqrt(magnitude);
    stepSize = stepSize * magnitude;
    /*
    if(stepSize < 0.00001 || magnitude < 0.00001){
      // Acceptable error no need to improve
      break;
    }
    */
    error = error * stepSize; //learningRate;

    weight = weight - error.transpose();
  }
  return weight;
}

int MLToolKit::test(std::vector<double>& y, std::vector<std::vector<double>>& data, Matrix w){
  int correct = 0;
  for(int i = 0; i < data.size(); i++){
    Matrix x(data[i]);
    Matrix mult = x * w;
    double element = mult.at(0)[0];
    if(element * y[i] > 0){
      correct++;
    }
  }

  return correct;
}

int MLToolKit::test(std::vector<double>& y, std::vector<std::vector<double>>& data){
  return test(y, data, weight);
}

int MLToolKit::testLogistic(std::vector<double>& y, std::vector<std::vector<double>>& data){
  double threshold = 0.1;
  int correct = 0;
  for(int i = 0; i < data.size(); i++){
    Matrix x(data[i]);
    double signal = (x * weight).at(0)[0];
    double out = LogisticFunction(signal);
    //std::cout << out << std::endl;
    int test = y[i];
    if(y[i] < 0){
      test = 0;
    }
    //std::cout << out << std::endl;
    if(fabs(out - test) < threshold){
      // tolerance of 0.1
      correct++;
    }
  }
  return correct;
}

