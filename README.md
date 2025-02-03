# Machine Learning Project: Digit Classification

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
  - [Perceptron](#perceptron)
  - [Non-linear Regression](#non-linear-regression)
  - [Digit Classification](#digit-classification)
  - [Dataset Classes](#dataset-classes)

## Introduction
This project implements machine learning models for different tasks, including perceptron-based classification, non-linear regression, and digit classification using neural networks.

## Installation
Ensure you have the necessary dependencies installed. We recommend using a Conda environment:

```bash
conda create --name ml_project python=3.8
conda activate ml_project
pip install numpy matplotlib torch torchvision
```

To verify installation, run:
```bash
python autograder.py --check-dependencies
```

## Project Structure
```
├── models.py               # Implements Perceptron, Regression, and Digit Classification Models
├── autograder.py           # Autograder script
├── backend.py              # Backend utilities (do not modify)
├── data/                   # Dataset for digit classification
├── README.md               # Project documentation
```

## Implementation Details

### Perceptron
- Implements a binary perceptron for classification.
- Key components:
  - `__init__(self, dimensions)`: Initializes weight parameters.
  - `run(self, x)`: Computes the dot product of weights and input.
  - `get_prediction(self, x)`: Returns +1 or -1 based on the computed value.
  - `train(self)`: Iteratively updates weights until 100% training accuracy is achieved.

### Non-linear Regression
- Implements a neural network to approximate `sin(x)` over `[-2π, 2π]`.
- Uses Mean Squared Error (MSE) loss function.
- Model is trained using gradient-based updates.

### Digit Classification
- Implements a neural network to classify handwritten digits.
- Uses cross-entropy loss for classification.
- Achieves 97%+ accuracy on the MNIST dataset.
- Uses dataset validation accuracy for training decisions.

### Dataset Classes
This project includes several dataset classes that help preprocess and handle different datasets:
- **Custom_Dataset**: Base dataset class used for handling various data inputs.
- **PerceptronDataset**: Generates synthetic data for training the perceptron model.
- **RegressionDataset**: Provides data points sampled from a sine function for regression.
- **DigitClassificationDataset**: Loads the MNIST dataset for handwritten digit classification.
- **LanguageIDDataset**: Handles language identification data for classification.

Each dataset class ensures proper data handling and visualization using Matplotlib.
