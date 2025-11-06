import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from StandardScalerNP import StandardScalerNP
from LinearRegressionNP2 import LinearRegressionNP2

def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]

def recover_original_coeffs(model_w, scaler_X, scaler_y):
    """Change standardized data into the original price scale"""
    w0 = model_w[0]
    w  = model_w[1:]
    sigma_y = scaler_y.scale_[0]
    mu_y    = scaler_y.mean_[0]
    sigma_X = scaler_X.scale_
    mu_X    = scaler_X.mean_
    beta = sigma_y * w / sigma_X                # hệ số từng feature (thang gốc)
    beta0 = mu_y + sigma_y * w0 - np.sum(beta * mu_X)  # intercept (thang gốc)
    return beta0, beta

def train_test_split_np(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# -------- Load data --------
houses = pd.read_csv(r"D:\Desktop\IMPORTANT_STUDY\Системы_искусственного_интеллекта\prak_1_california\California_Houses.csv")

# Explore data
print("Dataset shape:", houses.shape)
print("\nFirst 5 rows:")
print(houses.head())
print("\nMissing values:")
print(houses.isnull().sum())

y = houses.iloc[:, 0].values  # price
X = houses.iloc[:, 1:].values # features

# train/test split
X_train, X_test, y_train, y_test = train_test_split_np(
    X, y, test_size=0.2, random_state=42
)

# scale data 
scaler_X = StandardScalerNP()
scaler_y = StandardScalerNP()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# add column bias
X_train_b = add_bias(X_train_s)
X_test_b  = add_bias(X_test_s)

# take validation from train 
X_tr, X_val, y_tr, y_val = train_test_split_np(
    X_train_b, y_train_s, test_size=0.1, random_state=42
)

# -------- Train --------
model = LinearRegressionNP2(l2=1e-3)
model.fit(
    X_tr, y_tr,
    learning_rate=0.01,
    epochs=5000,
    validation_data=(X_val, y_val),
    patience=200,
    tol=1e-7,
    verbose=True 
)

# -------- Predict (scaled) -> inverse to original scale --------
y_pred_train_s = model.predict(X_train_b)
y_pred_test_s  = model.predict(X_test_b)

y_pred_train = scaler_y.inverse_transform(y_pred_train_s.reshape(-1, 1)).flatten()
y_pred_test  = scaler_y.inverse_transform(y_pred_test_s.reshape(-1, 1)).flatten()

# -------- Metrics on original scale --------
tr_mse, tr_mae, tr_rmse, tr_r2 = model.compute_metrics(y_train, y_pred_train)
te_mse, te_mae, te_rmse, te_r2 = model.compute_metrics(y_test,  y_pred_test)

print("\n=== Metrics on original scale ===")
print(f"[TRAIN] MSE={tr_mse:.3f} | MAE={tr_mae:.3f} | RMSE={tr_rmse:.3f} | R2={tr_r2:.4f}")
print(f"[TEST ] MSE={te_mse:.3f} | MAE={te_mae:.3f} | RMSE={te_rmse:.3f} | R2={te_r2:.4f}")

# -------- Coefficient on the root scale (to explain) --------
beta0, beta = recover_original_coeffs(model.weights, scaler_X, scaler_y)
feature_names = list(houses.columns[1:])
coef_table = pd.DataFrame({"feature": feature_names, "coef_original_scale": beta})
coef_table = coef_table.sort_values("coef_original_scale", key=lambda s: np.abs(s), ascending=False)
print("\n=== Features & Coefficients (original scale) ===")
print("Intercept (beta0):", beta0)
print(coef_table.to_string(index=False))

# -------- Plots --------
# 1) Learning curves
epochs_seen = np.arange(1, len(model.train_mse) + 1)
plt.figure(figsize=(9,5))
plt.plot(epochs_seen, model.train_mse, label="Train MSE (reg.)")
if len(model.val_mse) > 0:
    plt.plot(epochs_seen[:len(model.val_mse)], model.val_mse, label="Val MSE (reg.)")
plt.xlabel("Epoch")
plt.ylabel("MSE (with L2)")
plt.title("Learning Curves (Train / Validation)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2) Predicted vs Actual
plt.figure(figsize=(12,5))
# train
plt.subplot(1,2,1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
mn, mx = y_train.min(), y_train.max()
plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
plt.xlabel("Actual (Train)")
plt.ylabel("Predicted (Train)")
plt.title(f"Train: Predicted vs Actual (R2={tr_r2:.3f})")
# test
plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
mn, mx = y_test.min(), y_test.max()
plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
plt.xlabel("Actual (Test)")
plt.ylabel("Predicted (Test)")
plt.title(f"Test: Predicted vs Actual (R2={te_r2:.3f})")
plt.tight_layout()
plt.show()

# 3) Residual plot (Test)
residuals = y_test - y_pred_test
plt.figure(figsize=(9,5))
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted (Test)")
plt.ylabel("Residuals (y_true - y_pred)")
plt.title("Residuals vs Predicted (Test)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
