#include <iostream>
#include "MatrixMath.hpp"

double MatrixMath::matrixMult1D(std::vector<double>& matrixA, std::vector<double>& matrixB){
  if(matrixA.size() != matrixB.size()){
    throw "ERROR: Size of both vectors must be the same";
  }
  double output = 0.0;
  for(int i = 0; i < matrixA.size(); i++){
    output += matrixA[i] * matrixB[i];
  }

  return output;
}

std::vector<std::vector<double>> MatrixMath::matrixMult2D(std::vector<std::vector<double>>& matrixA, std::vector<std::vector<double>>& matrixB){
  std::vector<std::vector<double>> transposed = transpose2D(matrixB);
  // Invalid input
  if(matrixA[0].size() != transposed[0].size()){
    throw "ERROR: Row size of Matrix A must equal column size of Matrix B";
  }
  std::vector<std::vector<double>> output;
  // Begin multiplication
  for(int i = 0; i < matrixA.size(); i++){
    output.push_back(std::vector<double>());
    for(int j = 0; j < transposed.size(); j++){
      double element = matrixMult1D(matrixA[i], transposed[j]);
      output[i].push_back(element);
    }
  }

  return output;
}

std::vector<std::vector<double>> MatrixMath::add(std::vector<std::vector<double>>& matrixA, std::vector<std::vector<double>>& matrixB){
  if(matrixA.size() != matrixB.size()){
    throw "ERROR: Row size of matrix A does not match row size of matrix B";
  }
  std::vector<std::vector<double>> output;
  for(int i = 0; i < matrixA.size(); i++){
    if(matrixA[i].size() != matrixB[i].size()){
      throw "ERROR: Column size of both matrices must be the same";
    }
    output.push_back(std::vector<double>());
    for(int j = 0; j < matrixA[i].size(); j++){
      output[i].push_back(matrixA[i][j] + matrixB[i][j]);
    }
  }

  return output;
}

std::vector<std::vector<double>> MatrixMath::transpose2D(std::vector<std::vector<double>>& matrix){
  std::vector<std::vector<double>> output;
  // Apply transposition/rotate matrix
  for(int col = 0; col < matrix[0].size(); col++){
    output.push_back(std::vector<double>());
    for(int row = 0; row < matrix.size(); row++){
      output[col].push_back(matrix[row][col]);
    }
  }

  return output;
}

void MatrixMath::swap(std::vector<double>& vecA, std::vector<double>& vecB){
  std::vector<double> temp = vecA;
  for(int i = 0; i < vecA.size(); i++){
    vecA[i] = vecB[i];
    vecB[i] = temp[i];
  }
}

std::vector<std::vector<double>> MatrixMath::inverse(std::vector<std::vector<double>>& matrix){
  // Augment matrix with Identity matrix
  for(int i = 0; i < matrix.size(); i++){
    for(int j = 0; j < matrix.size(); j++){
      matrix[i].push_back(0);
    }
    matrix[i][i + matrix.size()] = 1;
  }
  
  // Gaussian Elimination
  int row = 0; // matrix row
  int col = 0; // matrix column
  while(row < matrix.size() && col < matrix[0].size()){
    // Find row with largest leading coefficient
    double max = matrix[row][col];
    int maxIndex = row;
    for(int i = row; i < matrix.size(); i++){
      if(matrix[i][col] > max){
        max = matrix[i][col];
        maxIndex = i;
      }
    }
    // Largest Coefficient is 0, nothing to do
    if(matrix[maxIndex][col] == 0){
      col++;
    }
    else{
      if(maxIndex != row){
        // Swap so that top row has the largest coefficient 
        swap(matrix[row], matrix[maxIndex]);
      }
      // Set leading coefficient to 1
      double ratio = 1.0 / matrix[row][col];
      // Apply ratio to rest of row
      for(int i = col; i < matrix[row].size(); i++){
        matrix[row][i] = matrix[row][i] * ratio;
      }
      // Solve rest of matrix
      for(int i = 0; i < matrix.size(); i++){
        if(i != row){
          // Calculate ratio
          ratio = matrix[i][col];
          // Zero the rest of the column under the pivot
          matrix[i][col] = 0;
          for(int j = col + 1; j < matrix[i].size(); j++){
            matrix[i][j] = matrix[i][j] - matrix[row][j]*ratio;
          }
        }
      }
      // Advance to next pivot
      row++;
      col++;
    }
  }

  // Get the inverted matrix
  std::vector<std::vector<double>> output;
  for(int i = 0; i < matrix.size(); i++){
    output.push_back(std::vector<double>());
    for(int j = matrix[i].size()/2; j < matrix[i].size(); j++){
      output[i].push_back(matrix[i][j]);
    }
  }

  return output;
}

double MatrixMath::error(std::vector<std::vector<double>>& data, std::vector<double>& weight, std::vector<double>& output){
  std::vector<std::vector<double>> w;
  w.push_back(weight);
  w = transpose2D(w);

  std::vector<std::vector<double>> out = matrixMult2D(data, w);
  out = transpose2D(out);

  std::vector<std::vector<double>> y;
  y.push_back(output);
  // Make -y
  for(int i = 0; i < y[0].size(); i++){
    y[0][i] = y[0][i] * -1;
  }
  std::vector<double> err = add(out, y)[0];
  double error = 0.0;
  for(int i = 0; i < err.size(); i++){
    error += err[i] * err[i];
  }
  error = error / data.size();

  return error;
}

std::vector<double> MatrixMath::regression(std::vector<std::vector<double>>& data, std::vector<double>& output){
  std::vector<std::vector<double>> weight;
  std::vector<std::vector<double>> transposed = transpose2D(data);
  std::vector<std::vector<double>> y;
  for(int i = 0; i < output.size(); i++){
    y.push_back(std::vector<double>());
    y[i].push_back(output[i]);
  }

  std::vector<std::vector<double>> mult = matrixMult2D(transposed, data);
  std::vector<std::vector<double>> inversed = inverse(mult);

  std::vector<std::vector<double>> pseudoInverse = matrixMult2D(inversed, transposed);
  weight = matrixMult2D(pseudoInverse, y);
  std::vector<double> out = transpose2D(weight)[0];

  return out;
}

