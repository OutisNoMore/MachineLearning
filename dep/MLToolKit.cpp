#include "MLToolKit.hpp"

MLToolKit::MLToolKit(std::string fileName){
  std::string line;
  std::ifstream dataFile(fileName);

  if(dataFile.is_open()){
    int i = 0;
    while(getline(dataFile, line)){
      dataX.push_back(std::vector<double>());
      while(line.size() > 0){
        int commaIndex = line.find(",");
        if(commaIndex == -1){
          if(line.find("veriscolor") == std::string::npos){
            output.push_back(1);
          } else{
            output.push_back(-1);
          }
          line = "";
        }
        else{
          double element = std::stod(line.substr(0, commaIndex));
          dataX[i].push_back(element);
          line = line.substr(commaIndex + 1);
        }
      }
      i++;
    }
    dataFile.close();
  }
  else{
    throw "ERROR: File not found!";
  }
}

void MLToolKit::Regression(){
  weight = MatrixMath::regression(dataX, output);
}

void MLToolKit::PerceptronLearning(){
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

void MLToolKit::test(std::string fileName, std::vector<double>& w){
  std::vector<std::vector<double>> testData;
  std::vector<double> testOutput;
  std::string line;
  std::ifstream dataFile(fileName);

  if(dataFile.is_open()){
    int i = 0;
    while(getline(dataFile, line)){
      testData.push_back(std::vector<double>());
      while(line.size() > 0){
        int commaIndex = line.find(",");
        if(commaIndex == -1){
          if(line.find("virginica") == std::string::npos){
            testOutput.push_back(-1);
          } else{
            testOutput.push_back(1);
          }
          line = "";
        }
        else{
          double element = std::stod(line.substr(0, commaIndex));
          testData[i].push_back(element);
          line = line.substr(commaIndex + 1);
        }
      }
      i++;
    }
    dataFile.close();
  }
  else{
    throw "ERROR: File not found!";
  }

  int correct = 0;
  for(int i = 0; i < testData.size(); i++){
    double element = MatrixMath::matrixMult1D(testData[i], w);
    if(element * output[i] > 0){
      correct++;
    }
  }

  std::cout << "Linear Regression Learning on Irises" << std::endl;;
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Total Test Samples:           " << testData.size() << std::endl;
  std::cout << "Total Classified Correctly:   " << correct << std::endl;
  std::cout << "Total Classified Incorrectly: " << testData.size() - correct << std::endl;
  std::cout << "Percent Correct:              " << (double)correct/testData.size() << std::endl;
  std::cout << "Weight used:                  ";
  for(int i = 0; i < w.size(); i++){
    std::cout << w[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
}

void MLToolKit::test(std::string fileName){
  test(fileName, weight);
}

