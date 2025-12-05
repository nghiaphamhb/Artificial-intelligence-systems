# MNIST Neural Network From Scratch 🧠✍️

This project implements a full neural-network training pipeline **without using PyTorch layers** such as `nn.Linear`, `nn.Conv2d`, `nn.MaxPool2d`, or `nn.Dropout`.
All components — Linear, ReLU, Dropout, Conv2d, MaxPool2D, Flatten, Softmax, and Adam — are **built manually** using only tensor operations + autograd.

---

## Features 🚀

### **✔ Manual Deep Learning Layers**

* Custom `Linear`
* Custom `Conv2d` (loop-based)
* Custom `MaxPool2D`
* `Dropout`, `ReLU`, `Flatten`
* `SoftmaxCrossEntropy`
* Custom `AdamOptimizer`

No neural-network modules from `torch.nn` are used.

### **✔ Manual MNIST Loader**

Reads raw `.idx` files using `struct`:

* 60,000 training images
* 10,000 test images
* 28×28 grayscale

### **✔ Four Models Implemented**

1. **MLP 1** — Basic Dense Network
2. **MLP 2** — Larger + Dropout
3. **CNN 1** — Custom Conv + Pool
4. **CNN 2** — Larger CNN

Each model trains separately and logs performance.

### **✔ Training Visualizations**

Plots for all models:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

This allows easy model comparison.

### **✔ Handwritten Digit Prediction ✏️➡️🔢**

A final model can:

* Accept an image drawn in Paint/GIMP
* Resize → grayscale → invert → normalize
* Predict the digit

### **Dataset**

Link to dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/code/data

---

## Why This Project? 🎯

To explore **how neural networks work internally**, without relying on high-level PyTorch layers.
It is a learning-focused implementation meant to teach:

* Tensor transformations
* Convolution mechanics
* Optimization
* Model comparison
* Practical evaluation

---

## Requirements 📦

```
Python 3.8+
PyTorch (tensor + autograd only)
NumPy
Matplotlib
```

---

## Structure 📁

```
/data           # MNIST idx files
notebooks/      # Training pipeline
README.md
main.ipynb
```
