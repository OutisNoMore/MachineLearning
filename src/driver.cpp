#include <iostream>
#include <string>
#include <fstream>
#include <map>

#include "MatrixMath.hpp"
#include "MLToolKit.hpp"

std::map<std::string, std::string> keys{
  {"admin.", "1"},
  {"unknown", "2"},
  {"unemployed", "3"},
  {"management", "4"},
  {"housemaid", "5"},
  {"entrepreneur", "6"},
  {"student", "7"},
  {"blue-collar", "8"},
  {"self-employed", "9"},
  {"retired", "10"},
  {"technician", "11"},
  {"services", "12"},
  {"married", "13"},
  {"divorced", "14"},
  {"single", "15"},
  {"secondary", "35"},
  {"primary", "36"},
  {"tertiary", "37"},
  {"yes", "16"},
  {"no", "17"},
  {"telephone", "18"},
  {"cellular", "19"},
  {"jan", "20"},
  {"feb", "21"},
  {"mar", "22"},
  {"apr", "23"},
  {"may", "24"},
  {"jun", "25"},
  {"jul", "26"},
  {"aug", "27"},
  {"sep", "28"},
  {"oct", "29"},
  {"nov", "30"},
  {"dec", "31"},
  {"other", "32"},
  {"failure", "33"},
  {"success", "34"},
  {"none", "0"},
};

void stats(std::vector<std::vector<double>>& data, std::vector<double>& w, int correct){
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Total Test Samples:           " << data.size() << std::endl;
  std::cout << "Total Classified Correctly:   " << correct << std::endl;
  std::cout << "Total Classified Incorrectly: " << data.size() - correct << std::endl;
  std::cout << "Percent Correct:              " << (double)correct/data.size() << std::endl;
  std::cout << "Weight used:                  ";
  for(int i = 0; i < w.size(); i++){
    if(i == w.size() - 1){
      std::cout << w[i];
    }
    else{
      std::cout << w[i] << ", ";
    }
  }
  std::cout << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
}

void readFile(std::string fileName, std::vector<double>& output, std::vector<std::vector<double>>& data){
  std::string line;
  std::ifstream dataFile(fileName);

  if(dataFile.is_open()){
    int i = 0;
    while(getline(dataFile, line)){
      data.push_back(std::vector<double>());
      while(line.size() > 0){
        int commaIndex = line.find(";");
        if(commaIndex == -1){
          if(line.find("no") == std::string::npos){
            output.push_back(1);
          } else{
            output.push_back(-1);
          }
          line = "";
        }
        else{
          std::string sub = line.substr(0, commaIndex);
          if(keys.count(sub) > 0){
            sub = keys[sub];
          }
          double element = std::stod(sub);
          data[i].push_back(element);
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

void printVector(std::vector<std::vector<double>> matrix){
  for(int i = 0; i < matrix.size(); i++){
    for(int j = 0; j < matrix[i].size(); j++){
      std::cout << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(){
  try{
/*  
  std::vector<std::vector<double>> matrix;
  matrix.push_back(std::vector<double>());
  matrix.push_back(std::vector<double>());
  matrix.push_back(std::vector<double>());
  matrix.push_back(std::vector<double>());
  matrix.push_back(std::vector<double>());
  matrix[0].push_back(5);
  matrix[0].push_back(2);
  matrix[0].push_back(1);
  matrix[0].push_back(4);
  matrix[0].push_back(6);
  
  matrix[1].push_back(9);
  matrix[1].push_back(4);
  matrix[1].push_back(2);
  matrix[1].push_back(5);
  matrix[1].push_back(2);
  
  matrix[2].push_back(11);
  matrix[2].push_back(5);
  matrix[2].push_back(7);
  matrix[2].push_back(3);
  matrix[2].push_back(9);

  matrix[3].push_back(5);
  matrix[3].push_back(6);
  matrix[3].push_back(6);
  matrix[3].push_back(7);
  matrix[3].push_back(2);

  matrix[4].push_back(7);
  matrix[4].push_back(5);
  matrix[4].push_back(9);
  matrix[4].push_back(3);
  matrix[4].push_back(3);

  std::vector<std::vector<double>> output = MatrixMath::inverse(matrix);
  printVector(output);

  std::vector<std::vector<double>> matrixA;
  std::vector<std::vector<double>> matrixB;

  matrixA.push_back(std::vector<double>());
  matrixA.push_back(std::vector<double>());

  matrixA[0].push_back(1);
  matrixA[0].push_back(2);
  matrixA[0].push_back(3);

  matrixA[1].push_back(4);
  matrixA[1].push_back(5);
  matrixA[1].push_back(6);

  matrixB.push_back(std::vector<double>());
  matrixB.push_back(std::vector<double>());
  matrixB.push_back(std::vector<double>());

  matrixB[0].push_back(7);
  matrixB[0].push_back(8);
  matrixB[1].push_back(9);
  matrixB[1].push_back(10);
  matrixB[2].push_back(11);
  matrixB[2].push_back(12);

  output = MatrixMath::matrixMult2D(matrixA, matrixB);
  printVector(output);

  matrixB.pop_back();
  matrixB[0].push_back(9);
  matrixB[1][0] = 10;
  matrixB[1][1] = 11;
  matrixB[1].push_back(12);
  output = MatrixMath::add(matrixA, matrixB);
  printVector(output);

  std::vector<double> testWeight;
  testWeight.push_back(1);
  testWeight.push_back(2);
  testWeight.push_back(3);
  std::vector<double> testOutput;
  testOutput.push_back(1);
  testOutput.push_back(2);
  //std::vector<double> w = MatrixMath::regression(matrixA, test);
  //for(int i = 0; i < w.size(); i++){
    //std::cout << w[i] << " ";
//  }
  std::cout << std::endl;
  MatrixMath::error(matrixA, testWeight, testOutput);
  std::vector<std::vector<double>> matrixA;
  for(int i = 0; i < 100; i++){
    matrixA.push_back(std::vector<double>());
  }
  int a = 0;
  for(int i = 0; i < matrixA.size(); i++){
    for(int j = 0; j < 5; j++){
      matrixA[i].push_back(a++);
    }
  }
  std::vector<std::vector<double>> out = MatrixMath::transpose2D(matrixA); 
  printVector(out);
  */
  std::vector<double> output;
  std::vector<std::vector<double>> data;
  readFile("data/bank-full.csv", output, data);
  MLToolKit test(output, data);
  std::vector<double> pocket = test.PocketLearning(); // weights
  std::vector<double> perceptron = test.PerceptronLearning();
  output.clear();
  data.clear();
  readFile("data/bank.csv", output, data);
  int correct = test.test(output, data);
  std::cout << "Linear Regression using the Perceptron Learning Algorithm on Bank Term Deposits" << std::endl;
  stats(data, perceptron, correct);
  std::cout << std::endl;

  correct = test.test(output, data, pocket);
  std::cout << "Linear Regression using the Pocket Learning Algorithm on Bank Term Deposits" << std::endl;
  stats(data, pocket, correct);
  }
  catch(const char* msg){
    std::cout << msg << std::endl;
  }

  return 0;
}
