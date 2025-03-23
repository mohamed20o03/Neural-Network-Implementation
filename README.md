# Neural Network Implementation

## Overview
This project implements a simple neural network from scratch using Python and NumPy. It includes a custom `Value` class for automatic differentiation and an `MLP` (Multi-Layer Perceptron) model for basic machine learning tasks.

## Features
✅ **Custom Computation Graph**: The `Value` class enables forward and backward computations with support for operations like addition, multiplication, exponentiation, and activation functions (ReLU, tanh, exp).  

✅ **Multi-Layer Perceptron (MLP)**: A simple feedforward neural network supporting training using gradient descent.  

✅ **Backpropagation**: The model computes gradients and updates weights using backpropagation.  

## Project Structure
### Neural-Network-Implementation
├──  Value.py          # Custom class for handling computation graph and automatic differentiation.
├──  NeuralNetworks.py # Implements the MLP model.
├──  train.py          # Training script.
└──  README.md         # Project documentation.

