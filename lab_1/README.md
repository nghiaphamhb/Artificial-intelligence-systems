# Lab 1  
### Linear & Logistic Regression (from scratch with NumPy & PyTorch)

This lab focuses on implementing two fundamental machine learning algorithms **from scratch** — without using high-level ML frameworks.  
Both **NumPy** (baseline) and **PyTorch (tensor-only, no autograd, no nn modules)** implementations are provided.

---

## 🔧 Tasks Overview

### **1. Linear Regression — California Housing Dataset**
- Implemented using:
  - **NumPy** (matrix multiplication + manual gradients)
  - **PyTorch** (tensor operations, manual gradient computation)
- Includes:
  - Feature preprocessing (scaling, bias term)
  - L2 regularization (Ridge)
  - Early stopping
  - Evaluation: MSE, MAE, RMSE, R²
  - Visualization:
    - Training/validation loss curves  
    - Predicted vs Actual plots  
    - Residual analysis  

---

### **2. Logistic Regression — Titanic Dataset**
- Implemented using:
  - **NumPy** (manual BCE loss + gradients)
  - **PyTorch** (manual sigmoid, BCE loss, gradient descent)
- Includes:
  - Data cleaning & preprocessing  
  - Standardization  
  - Stratified train/test split  
  - Evaluation: Accuracy, Precision, Recall, F1  
  - Visualization:
    - Loss reduction curve  
    - Metric curves (Accuracy/F1 over epochs)  

---

## 📁 Project Structure (example)

```bash 

/linear_numpy/
LinearRegressionNP2.py
/logistic_numpy/
LogisticRegressionNP.py
/linear_torch/
LinearRegressionTorch.py
/logistic_torch/
LogisticRegressionTorch.py
datasets/
California_Houses.csv
Titanic-Dataset.csv
notebooks/
linear_regression_colab.ipynb
logistic_regression_colab.ipynb
README.md

```

---

## 🚀 Key Learning Outcomes

- Understanding the **mathematical foundations** of linear and logistic regression  
- Implementing **gradient descent manually**  
- Using **PyTorch only as a tensor library**, without autograd or nn modules  
- Data preprocessing (cleaning, scaling, stratified splitting)  
- Building and interpreting:
  - Learning curves  
  - Error reduction plots  
  - R² evaluation  
  - Classification metrics (Precision/Recall/F1)

---

## 📊 Results Summary

- **Linear Regression (California Housing)**  
  - Achieved ~0.63–0.65 R² on test data  
  - Good convergence and stable learning behavior

- **Logistic Regression (Titanic)**  
  - F1 score and accuracy improve steadily during training  
  - Tensor-based PyTorch model matches NumPy baseline

---

## 📝 Conclusion

This lab demonstrates how to build two classic ML models **entirely from scratch**, reinforcing fundamental concepts such as matrix calculus, optimization, and evaluation. The PyTorch implementation highlights how neural-network frameworks can be used at a low level, focusing only on tensors and manual gradient computation.

