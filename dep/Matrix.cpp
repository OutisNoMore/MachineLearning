#include <iostream>
#include "Matrix.hpp"

Matrix::Matrix(std::vector<std::vector<double>>& matrix){
  this->matrix = matrix;
}

Matrix::Matrix(std::vector<double>& matrix){
  this->matrix.push_back(matrix);
}

std::vector<double>& Matrix::at(int i){
  if(i < 0 || i > matrix.size()){
    throw "ERROR: index out of bounds";
  }
  return matrix[i];
}

double Matrix::matrixMult1D(std::vector<double>& matrixA, std::vector<double>& matrixB){
  if(matrixA.size() != matrixB.size()){
    throw "ERROR: Size of both vectors must be the same";
  }
  double output = 0.0;
  for(int i = 0; i < matrixA.size(); i++){
    output += matrixA[i] * matrixB[i];
  }

  return output;
}

Matrix Matrix::operator*(Matrix& matrixB){
  if(this->matrix[0].size() != matrixB.matrix.size()){
    throw "ERROR: Row size of Matrix A must equal column size of Matrix B";
  }
  Matrix transposed = matrixB.transpose();
  Matrix output;
  // Begin multiplication
  for(int i = 0; i < this->matrix.size(); i++){
    output.matrix.push_back(std::vector<double>());
    for(int j = 0; j < transposed.matrix.size(); j++){
      double element = matrixMult1D(this->matrix[i], transposed.matrix[j]);
      output.matrix[i].push_back(element);
    }
  }

  return output;
}

Matrix Matrix::operator*(double coefficient){
  Matrix output;
  for(int i = 0; i < matrix.size(); i++){
    output.matrix.push_back(std::vector<double>());
    for(int j = 0; j < matrix[i].size(); j++){
      output.matrix[i].push_back(matrix[i][j] * coefficient);
    }
  }

  return output;
}

Matrix Matrix::operator+(Matrix& matrixB){
  if(this->matrix.size() != matrixB.matrix.size()){
    throw "ERROR: Row size of matrix A does not match row size of matrix B";
  }
  Matrix output;
  for(int i = 0; i < this->matrix.size(); i++){
    if(this->matrix[i].size() != matrixB.matrix[i].size()){
      throw "ERROR: Column size of both matrices must be the same";
    }
    output.matrix.push_back(std::vector<double>());
    for(int j = 0; j < this->matrix[i].size(); j++){
      output.matrix[i].push_back(this->matrix[i][j] + matrixB.matrix[i][j]);
    }
  }

  return output;
}

Matrix Matrix::operator-(Matrix matrixB){
  Matrix temp = matrixB;
  for(int i = 0; i < temp.matrix.size(); i++){
    for(int j = 0; j < temp.matrix[i].size(); j++){
      temp.matrix[i][j] *= -1;
    }
  }
  Matrix output = (*this) + temp;
  return output;
}

Matrix Matrix::transpose(){
  Matrix output;
  // Apply transposition/rotate matrix
  for(int col = 0; col < matrix[0].size(); col++){
    output.matrix.push_back(std::vector<double>());
    for(int row = 0; row < matrix.size(); row++){
      output.matrix[col].push_back(matrix[row][col]);
    }
  }

  return output;
}

void Matrix::swap(std::vector<double>& vecA, std::vector<double>& vecB){
  std::vector<double> temp = vecA;
  for(int i = 0; i < vecA.size(); i++){
    vecA[i] = vecB[i];
    vecB[i] = temp[i];
  }
}

Matrix Matrix::inverse(){
  if(this->matrix.size() != this->matrix[0].size()){
    throw "ERROR: Matrix must be square"; 
  }
  Matrix output = *this;
  // Augment matrix with Identity matrix
  for(int i = 0; i < output.matrix.size(); i++){
    if(matrix[i].size() != matrix.size()){
      throw "ERROR: Matrix must be square";
    }
    for(int j = 0; j < output.matrix.size(); j++){
      output.matrix[i].push_back(0);
    }
    output.matrix[i][i + output.matrix.size()] = 1;
  }
  
  // Gaussian Elimination
  int row = 0; // matrix row
  int col = 0; // matrix column
  while(row < output.matrix.size() && col < output.matrix[0].size()){
    // Find row with largest leading coefficient
    double max = output.matrix[row][col];
    int maxIndex = row;
    for(int i = row; i < output.matrix.size(); i++){
      if(output.matrix[i][col] > max){
        max = output.matrix[i][col];
        maxIndex = i;
      }
    }
    // Largest Coefficient is 0, nothing to do
    if(output.matrix[maxIndex][col] == 0){
      col++;
    }
    else{
      if(maxIndex != row){
        // Swap so that top row has the largest coefficient 
        swap(output.matrix[row], output.matrix[maxIndex]);
      }
      // Set leading coefficient to 1
      double ratio = 1.0 / output.matrix[row][col];
      // Apply ratio to rest of row
      for(int i = col; i < output.matrix[row].size(); i++){
        output.matrix[row][i] = output.matrix[row][i] * ratio;
      }
      // Solve rest of matrix
      for(int i = 0; i < output.matrix.size(); i++){
        if(i != row){
          // Calculate ratio
          ratio = output.matrix[i][col];
          // Zero the rest of the column under the pivot
          output.matrix[i][col] = 0;
          for(int j = col + 1; j < output.matrix[i].size(); j++){
            output.matrix[i][j] = output.matrix[i][j] - output.matrix[row][j]*ratio;
          }
        }
      }
      // Advance to next pivot
      row++;
      col++;
    }
  }

  // Get the inverted matrix
  for(int i = 0; i < output.matrix.size(); i++){
    output.matrix[i].erase(output.matrix[i].begin(), output.matrix[i].begin() + output.matrix[i].size()/2);
  }

  return output;
}

std::string Matrix::toString(){
  std::string output = "";
  for(int i = 0; i < this->matrix.size(); i++){
    for(int j = 0; j < this->matrix[i].size(); j++){
      output += std::to_string(matrix[i][j]) + " ";
    }
    output += "\n";
  }
  return output;
}

double Matrix::squaredMagnitude(){
  double output = 0.0;
  for(int i = 0; i < matrix.size(); i++){
    for(int j = 0; j < matrix[i].size(); j++){
      output += matrix[i][j] * matrix[i][j];
    }
  }
  return output;
}
