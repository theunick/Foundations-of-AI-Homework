# Foundations of Artificial Intelligence Homework

Machine learning algorithms implemented from scratch for the **Fondamenti di Intelligenza Artificiale** course (A.Y. 2023/2024).  
**Bachelor of Science in Ingegneria Informatica e Automatica**

## Student
**Nicolas Leone** - Student ID: 1986354  

## Overview

This project demonstrates the implementation of fundamental machine learning algorithms **entirely from scratch** using only NumPy and Pandas. The notebook includes comprehensive hyperparameter tuning, performance analysis, and validation against scikit-learn implementations.

**Dataset**: Obesity level estimation with 2,111 records and 17 features (eating habits, physical condition, demographics).

---

## 📊 Regression Implementations

### 1. Linear Regression (Matrix-Based)
- **Closed-form solution** using the normal equation
- Direct computation: `β = (X^T X)^(-1) X^T y`
- Pseudoinverse implementation with NumPy
- Instant training, optimal for linear relationships

### 2. Mini-Batch Stochastic Gradient Descent
- **Configurable batch sizes** (16, 32, 64) for flexible optimization
- Linear weight updates over multiple epochs
- Gradient clipping to prevent numerical instability
- **Grid search** over: learning rate, epochs, batch size
- Significant performance improvement over full-batch GD

### 3. Neural Network (Feed-Forward)
- **Multi-layer architecture** with customizable hidden layers
- **Sigmoid activation** for hidden layers
- **Backpropagation** algorithm for gradient computation
- Forward propagation with matrix operations
- **Flexible hyperparameters**: 
  - Number of neurons per layer (`k`)
  - Number of hidden layers
  - Learning rate and epochs
- Complete gradient descent training loop
- MSE tracking across epochs for convergence analysis

**Result**: Neural Network achieved the **lowest RMSE** among regression models.

---

## 🎯 Classification Implementations

### 1. Decision Tree (Entropy-Based)
- **Information gain** calculation for feature selection
- Recursive tree building with custom `Node` class
- **Entropy** as impurity measure: `-Σ(p * log₂(p))`
- Binary splitting on normalized features (threshold = 0.5)
- Max depth control to prevent overfitting
- Leaf node classification with majority class

### 2. Logistic Regression
- **Sigmoid function** for probability estimation: `σ(z) = 1/(1 + e^(-z))`
- **Gradient descent** optimization with iterative weight updates
- Gradient computation: `∇ = X^T(σ(Xw) - y) / m`
- **Grid search** over learning rates and iterations
- Accuracy tracking per iteration
- Overflow protection with gradient clipping

### 3. K-Nearest Neighbors (KNN)
- **Euclidean distance** computation for all training points
- **Majority voting** among k nearest neighbors
- Efficient NumPy vectorization for distance calculations
- **Comprehensive grid search**: k values from 1 to 30
- No training phase (lazy learning)
- Distance sorting with `np.argsort()`

**Result**: KNN achieved **~100% accuracy** with robust, stable performance.

---

## 🔧 Technical Implementation Details

### Data Preprocessing
- **Categorical encoding**: String attributes converted to numeric (one-hot style)
- **Normalization**: Z-score standardization (mean=0, std=1) for all features
- **Train/Test split**: 90/10 stratified split
- **Bias term insertion** for linear models
- Three prepared datasets: regression (with bias), classification (raw), classification (normalized)

### Hyperparameter Tuning
- **Custom grid search** implementations for each algorithm
- Exhaustive search over all hyperparameter combinations
- Validation-based selection (RMSE for regression, accuracy for classification)
- Performance visualization with matplotlib
- Best parameters tracked and reported

### Validation Strategy
- All custom implementations **validated against scikit-learn**
- Side-by-side performance comparison
- Metric parity verification (RMSE, accuracy)
- Ensures correctness of from-scratch implementations

---

## 🎨 Key Features

- ✅ **Zero ML frameworks** for model building (NumPy/Pandas only)
- ✅ **Complete mathematical implementations** of all algorithms
- ✅ **Extensive hyperparameter tuning** with grid search
- ✅ **Performance visualizations** (accuracy curves, RMSE plots)
- ✅ **Scikit-learn validation** for correctness verification
- ✅ **Critical analysis** comparing algorithm strengths/weaknesses
- ✅ **Clean, documented code** with extensive comments

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn (for comparison only)
```

## Notebook Execution

The notebook runs end-to-end without errors. Execute cells sequentially from top to bottom.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theunick/Homework-IA/blob/main/Homework_di_Nicolas_Leone_Fondamenti_di_IA_AA_2023_24.ipynb)

## Results Summary

- **Best Regression Model**: Neural Network (lowest RMSE)
- **Best Classification Model**: KNN (~100% accuracy, robust performance)
- All implementations validated against scikit-learn baselines

---

**Submission Deadline**: May 31st, 2024
