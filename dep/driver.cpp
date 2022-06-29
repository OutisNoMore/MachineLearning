#include <iostream>

#include "Matrix.hpp"

int main(){
  try{
    std::vector<std::vector<double>> test{
      {1, 2, 3, 4, 5},
      {6, 7, 8, 9, 0}
    };
    std::vector<std::vector<double>> test2{
      {9, 8, 7, 6, 5},
      {4, 3, 2, 1, 10}
    };

    // Test Construction
    Matrix matrixA(test);
    Matrix matrixB(test2);
    std::cout << matrixA.toString() << std::endl;
    std::cout << matrixB.toString() << std::endl;

    // Test Addition
    Matrix add = matrixA + matrixB;
    std::cout << add.toString() << std::endl;

    // Test Substract
    Matrix sub = matrixA - matrixB;
    std::cout << sub.toString() << std::endl;

    // Test multiplication
    std::vector<std::vector<double>> test3{
      {1, 2},
      {3, 4},
      {5, 6},
      {7, 8},
      {9, 0}
    };
    Matrix matrixC(test3);
    std::cout << matrixC.toString() << std::endl;
    Matrix mult = matrixA * matrixC;
    std::cout << mult.toString() << std::endl;

    // Test Transpose
    Matrix trans = matrixA.transpose();
    std::cout << trans.toString() << std::endl;

    // Test Inverse
    std::vector<std::vector<double>> test4{
      {1.5, 2.6, 3, 4, 5},
      {3.4, 4.7, 8, 9, 0},
      {6, 3, 5, 2, 1, 7},
      {1, 4, 2, 5, 0},
      {9, 0, 8, 2, 3}
    };
    Matrix inverse(test4);
    Matrix inv = inverse.inverse();
    std::cout << inv.toString() << std::endl;

    return 0;

  }
  catch (const char* msg){
    std::cout << msg << std::endl;
  }
}
