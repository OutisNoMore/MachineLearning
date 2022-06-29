#include <iostream>
#include <string>
#include <fstream>
#include <map>

#include "Matrix.hpp"
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

void stats(std::vector<std::vector<double>>& data, Matrix& w, int correct){
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Total Test Samples:           " << data.size() << std::endl;
  std::cout << "Total Classified Correctly:   " << correct << std::endl;
  std::cout << "Total Classified Incorrectly: " << data.size() - correct << std::endl;
  std::cout << "Percent Correct:              " << (double)correct/data.size() << std::endl;
  std::cout << "Weight used:                  " << w.toString() << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
}

void readFile(std::string fileName, std::vector<double>& output, std::vector<std::vector<double>>& data){
  std::string line;
  std::ifstream dataFile(fileName);

  if(dataFile.is_open()){
    int i = 0;
    while(getline(dataFile, line)){
      data.push_back(std::vector<double>());
      data[i].push_back(1.0);
      while(line.size() > 0){
        int commaIndex = line.find(",");
        if(commaIndex == -1){
          if(line.find("versicolor") == std::string::npos){
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
    std::vector<double> output;
    std::vector<std::vector<double>> data;
    // Test Logistic Regression methods
    readFile("data/iris-train.data", output, data);
    MLToolKit test(output, data);
    Matrix logistic = test.LogisticRegression();
    output.clear();
    data.clear();
    readFile("data/iris-test.data", output, data);
    int correct = test.testLogistic(output, data);
    std::cout << "Logistic Regression on iris dataset" << std::endl;
    stats(data, logistic, correct);

    /*
    // Test Linear regression methods
    readFile("data/bank-full.csv", output, data);
    MLToolKit test(output, data);
    Matrix perceptron = test.PerceptronLearning();
    Matrix pocket = test.PocketLearning(); // weights
    output.clear();
    data.clear();
    readFile("data/bank.csv", output, data);
    int correct = test.test(output, data, perceptron);
    std::cout << "Linear Regression using the Perceptron Learning Algorithm on Iris" << std::endl;
    stats(data, perceptron, correct);
    std::cout << std::endl;
    int correct = test.test(output, data, pocket);
    std::cout << "Linear Regression using the Pocket Learning Algorithm on Iris" << std::endl;
    stats(data, pocket, correct);
    */
  }
  catch(const char* msg){
    std::cout << msg << std::endl;
  }

  return 0;
}
