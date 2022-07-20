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
  Matrix identity = mult.identity(); 
  identity = identity * 10; // lambda 0.1
  mult = mult + identity;    // Regularization
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
  double stepSize = 0.0001; // step size of SGD
  // Maximum of 10,000 loops
  for(int i = 0; i < 100000; i++){
    Matrix error = GradientError(); // Calculate error
    double magnitude = error.squaredMagnitude(); // In sample error
    magnitude = sqrt(magnitude);
    double r = stepSize * magnitude;
    
    if(magnitude < 0.01 || stepSize < 0.00001){
      // Acceptable error no need to improve
      break;
    }
    
    error = error * r;
    //weight = weight * (1 - 2*r*(100/dataX.size())); // Regularization
    weight = weight - error.transpose();
  }
  return weight;
}

double MLToolKit::ThetaFunction(double signal){
  double theta = exp(signal) - exp(-1*signal);
  double theta2 = exp(signal) + exp(-1*signal);

  return theta/theta2;
}

std::vector<std::vector<std::vector<double>>> MLToolKit::NeuralNetwork(){
  srand(time(NULL));
  std::vector<std::vector<std::vector<double>>> weights; // Weights for Neurons
  std::vector<std::vector<double>> NeuralNetwork;        // 2D Grid of Neural Network
  std::vector<std::vector<double>> deltas;               // Deltas for weight adjustment
  // Initialize Neural Network
  NeuralNetwork.push_back(std::vector<double>());
  NeuralNetwork.push_back(std::vector<double>());
  // Number of nodes should be between 0 and number of nodes in previous layer
  int nodes = ((2.0/3.0) * dataX.at(0).size()) + 1;
  NeuralNetwork[1].push_back(1);
  for(int i = 0; i < nodes; i++){
    NeuralNetwork[1].push_back(0);
  }
  // Initialize weights
  weights.push_back(std::vector<std::vector<double>>());
  // First Layer weights
  for(int i = 0; i < NeuralNetwork[1].size() - 1; i++){
    weights[0].push_back(std::vector<double>());
    for(int j = 0; j < dataX.at(0).size(); j++){
      // Randomly initialize weights
      double w = (rand() % 100/1000.0);
      weights[0][i].push_back(w);
    }
  }
  // Second Layer weights
  weights.push_back(std::vector<std::vector<double>>());
  weights[1].push_back(std::vector<double>());
  for(int i = 0; i < NeuralNetwork[1].size(); i++){
    double w = (rand() % 100/1000.0);
    weights[1][0].push_back(w);
  }
  // Initialize delta values
  deltas.push_back(std::vector<double>());
  deltas.push_back(std::vector<double>());
  deltas[1].push_back(0);
  for(int i = 0; i < weights[0].size(); i++){
    deltas[0].push_back(0);
  }
  // Begin training Neural Network
  double stepSize = 0.00001;
  for(int i = 0; i < 1000000; i++){
    int stoData = rand() % dataX.size();  // Randomly choose Data for stochastic
    NeuralNetwork[0] = dataX.at(stoData); // Get input layer
    Matrix dat(NeuralNetwork[0]); 
    // Forward Propagation of hidden layer
    for(int k = 1; k < NeuralNetwork[1].size(); k++){
      Matrix wt(weights[0][k - 1]);
      wt = wt.transpose();
      Matrix x = dat * wt;
      double signal = x.at(0)[0];
      signal = ThetaFunction(signal);
      if(std::isnan(signal)){
        std::cerr << "ERROR: not a number 1!\n" ;
        std::cerr << wt.toString() << "\n";
        std::cerr << dat.toString() << "\n";
        std::cerr << x.toString() << "\n";
        throw "Not a Number!";
      }
      NeuralNetwork[1][k] = signal;
    }
    // Calculate output layer
    Matrix dat2(NeuralNetwork[1]); // Get Hidden layer
    Matrix wt(weights[1][0]);      // Get weights for output
    wt = wt.transpose();
    Matrix x = dat2 * wt; // Calculate output
    double out = x.at(0)[0];
    out = ThetaFunction(out);
    // Begin Backwards Propagation
    double delta = 1 - (out * out);
    if(std::isnan(delta)){
      std::cerr << "ERROR: not a number 2!\n" ;
      std::cerr << wt.toString() << "\n";
      std::cerr << dat2.toString() << "\n";
      std::cerr << delta << " " << out << std::endl;
      throw "Not a Number!";
    }
    //std::cout << delta << " " << out << std::endl;
    delta = 2 * delta * (out - output.at(stoData)[0]);
    //std::cout << delta * (out - output.at(stoData)[0]) << std::endl;
    deltas[1][0] = delta;
    //std::cout << deltas[1][0] << " " << delta << std::endl;
    for(int j = deltas.size() - 2; j >= 0; j--){
      for(int k = 1; k < deltas[j].size(); k++){
        delta = 0;
        for(int l = 0; l < deltas[j + 1].size(); l++){
          delta += deltas[j + 1][l] * weights[j+1][l][k];
          //std::cout << deltas[j + 1][l] << " " << weights[j+1][l][k] << std::endl;
        }
        //std::cout << delta << std::endl;
        deltas[j][k-1] = delta * (1 - NeuralNetwork[j+1][k] * NeuralNetwork[j+1][k]);
      }
    }
    // Update weights
    for(int j = 0; j < weights.size(); j++){
      for(int k = 0; k < weights[j].size(); k++){
        for(int l = 0; l < weights[j][k].size(); l++){
        //std::cout << deltas[j][k] << " " << weights[j][k][l] << std::endl;
          weights[j][k][l] = weights[j][k][l] - stepSize * NeuralNetwork[j][l] * deltas[j][k];
        }
      }
    }
    // TODO - Stop with error validation
  }
  return weights;
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
  double threshold = 0.5;
  int correct = 0;
  for(int i = 0; i < data.size(); i++){
    Matrix x(data[i]);
    double signal = (x * weight).at(0)[0];
    double out = LogisticFunction(signal);
    int test = y[i];
    if(y[i] < 0){
      test = 0;
    }
    std::cout << out << std::endl;
    if(fabs(out - test) < threshold){
      // tolerance of 0.1
      correct++;
    }
  }
  return correct;
}

int MLToolKit::testNeural(std::vector<double>& y, std::vector<std::vector<double>>& data, std::vector<std::vector<std::vector<double>>>& weights){
  double threshold = 0.5;
  int correct = 0;
  // Create Neural Network
  std::vector<std::vector<double>> NeuralNetwork;
  NeuralNetwork.push_back(std::vector<double>());
  NeuralNetwork.push_back(std::vector<double>());
  NeuralNetwork[1].push_back(1);
  for(int i = 0; i < weights[0].size(); i++){
    NeuralNetwork[1].push_back(0);
  }
  // Begin testing
  for(int i = 0; i < data.size(); i++){
    NeuralNetwork[0] = data[i]; // Input Layer
    Matrix x(NeuralNetwork[0]);
    for(int j = 0; j < weights[0].size(); j++){
      Matrix wt(weights[0][j]);
      wt = wt.transpose();
      Matrix out = x * wt;
      double signal = out.at(0)[0];
      signal = ThetaFunction(signal);
      NeuralNetwork[1][j + 1] = signal;
    }
    // Get output layer
    Matrix wt(weights[1][0]);
    wt = wt.transpose();
    x = Matrix{NeuralNetwork[1]};
    Matrix out = x * wt;
    double signal = out.at(0)[0];
    signal = ThetaFunction(signal);
    signal = fabs(signal);
    std::cout << signal << std::endl;
    // Compare against output
    int test = y[i];
    if(y[i] < 0){
      test = 0;
    }
    //std::cout << signal << " " << test << std::endl;
    if(fabs(signal - test) < threshold){
      // Threshold of 0.5 for correct
      correct++;
    }
  }
  return correct;
}

