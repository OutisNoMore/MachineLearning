#include <iostream>

#include "MatrixMath.hpp"
#include "MLToolKit.hpp"

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
  MLToolKit test("data/iris-train.data");
  test.PerceptronLearning();
  std::vector<double> pocket = test.PocketLearning();
  test.test("data/iris-test.data", pocket);
  //test.Regression();
  test.test("data/iris-test.data");
  }
  catch(const char* msg){
    std::cout << msg << std::endl;
  }

  return 0;
}
