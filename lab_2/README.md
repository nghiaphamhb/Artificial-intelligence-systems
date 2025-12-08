

# 🧪 Lab 2 — K-Nearest Neighbors (KNN), Regularization, and Model Comparison

### *Artificial Intelligence Systems – Laboratory Work #2*

---

## 📌 1. Objectives of the Lab

This lab consists of two major parts.

---

### **✔ Part 1 — Extension of Lab 1 (Linear & Logistic Regression)**

You must:

* Reuse the **Linear Regression** and **Logistic Regression** models implemented from scratch in Lab 1.
* Add three types of regularization:

  * **L1 (Lasso)**
  * **L2 (Ridge)**
  * **ElasticNet**
* Train models using all four configurations:

  * No regularization
  * L1
  * L2
  * ElasticNet
* Plot **four curves on a single plot** showing:

  * Loss decrease over epochs
  * Metric improvement (R² for regression, Accuracy/F1 for classification)

This part evaluates understanding of optimization, regularization, overfitting, and training dynamics.

---

### **✔ Part 2 — Implementing K-Nearest Neighbors (KNN) From Scratch**

You must implement the KNN algorithm using:

* **NumPy (basic version)**
* **PyTorch tensors (advanced version)**
  → *No high-level torch APIs (no nn.Module, no KDTree, no cdist). Only pure tensor operations.*

Your implementation must include:

```python
def __init__(...)   # model initialization
def fit(...)        # store training data
def predict(...)    # compute distances + KNN prediction
```

Must support **four distance metrics**:

| Metric    | Description           |
| --------- | --------------------- |
| Euclidean | L2 norm               |
| Manhattan | L1 norm               |
| Minkowski | General Lp norm       |
| Cosine    | 1 − cosine similarity |

You will apply KNN to both:

1. **Classification:** Fashion-MNIST
2. **Regression:** US Health Insurance dataset

---

## 📚 2. Datasets

### 🧵 Fashion-MNIST — Classification

* 60,000 training images
* 10,000 test images
* 28×28 grayscale
* 10 clothing categories

Used to test KNN classification accuracy with different metrics.

---

### 💵 US Health Insurance — Regression

Target variable: **charges**
Features: age, sex, bmi, children, smoker, region

Used to evaluate KNN regression performance.

---

## 🛠️ 3. Implementation Details

### ✔ **Part 1 — Linear & Logistic Regression with Regularization**

Implemented using **PyTorch tensors**, but:

* **No nn.Linear**
* **No autograd (gradient computed manually)**
* **No optimizers like Adam**

Regularized loss function:

[
L = MSE + \lambda (\alpha |w|_1 + (1 - \alpha)|w|_2^2)
]

You must plot:

* Training loss curves (4 lines)
* Validation loss curves (optional, but recommended)
* Training + validation metrics (R² or Accuracy/F1)

---

### ✔ **Part 2 — KNN Implementation (NumPy → PyTorch)**

#### NumPy version

Uses KDTree or cdist for distance computation (simple baseline).

#### PyTorch version (**required for full score**)

Distances computed manually:

```python
diff = x_batch.unsqueeze(1) - X_train.unsqueeze(0)
dist = (diff**2).sum(dim=2)
```

Supports batch processing to avoid memory overflow.

Runs on:

* **CPU**, or
* **GPU** automatically (`device="cuda"` if available)

---

## 📊 4. Evaluation Metrics

### **Classification (Fashion-MNIST)**

You must report:

* **Accuracy**
* **Precision (weighted)**
* **Recall (weighted)**
* **F1-score**
* **Confusion matrix (heatmap)**

Each metric should be evaluated for all four distance functions.

---

### **Regression (Insurance dataset)**

You must report:

* MSE
* RMSE
* MAE
* Median Absolute Error
* R² Score
* Explained Variance Score

Additionally:

* Plot **MSE across k = 1 → 15**
  → Helps analyze underfitting/overfitting trade-offs.

---

## 📁 5. Suggested Folder Structure

```
├── README.md
├── regularization/
│   ├── linear_regression_torch.py
│   ├── logistic_regression_torch.py
│   └── plots/
├── knn/
│   ├── knn_numpy.py
│   ├── knn_torch_classifier.py
│   ├── knn_torch_regressor.py
│   └── results/
├── datasets/
│   ├── fashion-mnist_train.csv
│   ├── fashion-mnist_test.csv
│   ├── insurance.csv
└── notebooks/
    ├── Lab2_KNN_Classification.ipynb
    ├── Lab2_KNN_Regression.ipynb
    └── Lab2_Regularization.ipynb
```

---

## 🧠 6. Conclusion

At the end of this lab, students should be able to:

* Understand and implement L1, L2, and ElasticNet regularization.
* Build Linear/Logistic Regression from scratch in PyTorch using manual gradients.
* Implement KNN without relying on KDTree/cdist — using only tensor math.
* Evaluate both classification and regression tasks with appropriate metrics.
* Analyze how distance metrics and the value of K affect model performance.

