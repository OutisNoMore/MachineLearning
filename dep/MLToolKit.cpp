#include "MLToolKit.hpp"

MLToolKit::MLToolKit(std::vector<double>& y, std::vector<std::vector<double>>& data){
  output = y;
  dataX = data;
}

void MLToolKit::Regression(){
  weight = MatrixMath::regression(dataX, output);
}

std::vector<double> MLToolKit::PerceptronLearning(){
  // Get initial weights using Linear Regression
  Regression();
  for(int i = 0; i < dataX.size(); i++){
    std::vector<double> x = dataX[i];
    double hyp = MatrixMath::matrixMult1D(weight, x);
    if(hyp * output[i] < 0){
      // Mis-classified data
      for(int j = 0; j < x.size(); j++){
        x[j] = x[j] * output[i];
      }
      std::vector<std::vector<double>> weight2D;
      weight2D.push_back(weight);
      std::vector<std::vector<double>> x2D;
      x2D.push_back(x);
      weight = MatrixMath::add(weight2D, x2D)[0];
    }
  }
  return weight;
}

std::vector<double> MLToolKit::PocketLearning(){
  Regression();
  double err = MatrixMath::error(dataX, weight, output);
  std::vector<double> pocket = weight;
  for(int i = 0; i < dataX.size(); i++){
    std::vector<double> x = dataX[i];
    double hyp = MatrixMath::matrixMult1D(weight, x);
    if(hyp * output[i] < 0){
      // Mis-classified data
      for(int j = 0; j < x.size(); j++){
        x[j] = x[j] * output[i];
      }
      std::vector<std::vector<double>> weight2D;
      weight2D.push_back(weight);
      std::vector<std::vector<double>> x2D;
      x2D.push_back(x);
      weight = MatrixMath::add(weight2D, x2D)[0]; // Re-calibrated weight
      double test = MatrixMath::error(dataX, weight, output);
      if(test < err){
        pocket = weight;
        err = test;
      }
    }
  }
  return pocket;
}

int MLToolKit::test(std::vector<double>& y, std::vector<std::vector<double>>& data, std::vector<double>& w){
  std::vector<std::vector<double>> testData = data;
  std::vector<double> testOutput = y;

  int correct = 0;
  for(int i = 0; i < testData.size(); i++){
    double element = MatrixMath::matrixMult1D(testData[i], w);
    if(element * output[i] > 0){
      correct++;
    }
  }

  return correct;
}

int MLToolKit::test(std::vector<double>& y, std::vector<std::vector<double>>& data){
  return test(y, data, weight);
}

