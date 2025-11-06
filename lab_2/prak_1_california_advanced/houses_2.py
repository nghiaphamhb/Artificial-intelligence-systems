import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from StandardScalerNP import StandardScalerNP
from LinearRegressionNP2 import LinearRegressionNP2

# -------- Utility functions --------

# Thêm bias (cột 1) vào X
def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]

# Khôi phục các hệ số của mô hình về thang đo ban đầu
def recover_original_coeffs(model_w, scaler_X, scaler_y):
    w0 = model_w[0]; w = model_w[1:]
    sigma_y = scaler_y.scale_[0]; mu_y = scaler_y.mean_[0]
    sigma_X = scaler_X.scale_;    mu_X = scaler_X.mean_
    beta = sigma_y * w / sigma_X
    beta0 = mu_y + sigma_y * w0 - np.sum(beta * mu_X)
    return beta0, beta

# Chia tập train/test
def train_test_split_np(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]; train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# -------- Load data --------
houses = pd.read_csv(r"D:\Desktop\IMPORTANT_STUDY\Системы_искусственного_интеллекта\lab_2\prak_1_california_advanced\California_Houses.csv")
print("Dataset shape:", houses.shape)
print(houses.head())
print(houses.isnull().sum())

y = houses.iloc[:, 0].values  # price
X = houses.iloc[:, 1:].values # features

# split
X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, random_state=42)

# scale
scaler_X = StandardScalerNP()
scaler_y = StandardScalerNP()
X_train_s = scaler_X.fit_transform(X_train); X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# bias
X_train_b = add_bias(X_train_s); X_test_b = add_bias(X_test_s)

# validation từ train
X_tr, X_val, y_tr, y_val = train_test_split_np(X_train_b, y_train_s, test_size=0.1, random_state=42)

# -------- Train (chọn 1 trong 3) --------
# L2 (ridge):
model = LinearRegressionNP2(penalty='l2', lamda=1e-3)
# L1:
# model = LinearRegressionNP2(penalty='l1', lamda=5e-4)
# ElasticNet:
# model = LinearRegressionNP2(penalty='elasticnet', lamda=1e-3, alpha=0.3)

model.fit(
    X_tr, y_tr,
    learning_rate=0.01,
    epochs=600,
    validation_data=(X_val, y_val),
    patience=60,
    tol=1e-7,
    verbose=True
)

# -------- Predict (scaled) -> back original scale --------
y_pred_train_s = model.predict(X_train_b); y_pred_test_s = model.predict(X_test_b)
y_pred_train = scaler_y.inverse_transform(y_pred_train_s.reshape(-1, 1)).flatten()
y_pred_test  = scaler_y.inverse_transform(y_pred_test_s.reshape(-1, 1)).flatten()

# -------- Metrics on original scale --------
tr_mse, tr_mae, tr_rmse, tr_r2 = model.compute_metrics(y_train, y_pred_train)
te_mse, te_mae, te_rmse, te_r2 = model.compute_metrics(y_test,  y_pred_test)
print("\n=== Metrics on original scale ===")
print(f"[TRAIN] MSE={tr_mse:.3f} | MAE={tr_mae:.3f} | RMSE={tr_rmse:.3f} | R2={tr_r2:.4f}")
print(f"[TEST ] MSE={te_mse:.3f} | MAE={te_mae:.3f} | RMSE={te_rmse:.3f} | R2={te_r2:.4f}")

# -------- Coefficients on original scale --------
beta0, beta = recover_original_coeffs(model.weights, scaler_X, scaler_y)
feature_names = list(houses.columns[1:])
coef_table = pd.DataFrame({"feature": feature_names, "coef_original_scale": beta})
coef_table = coef_table.sort_values("coef_original_scale", key=lambda s: np.abs(s), ascending=False)
print("\n=== Features & Coefficients (original scale) ===")
print("Intercept (beta0):", beta0)
print(coef_table.to_string(index=False))

# -------- SINGLE PLOT: 4 lines (Train/Val Loss, Val R2, Val MAE) --------
# Chuẩn hoá loss/MAE về giá trị đầu tiên để cùng trục
eps = 1e-12
epochs_seen = np.arange(1, len(model.train_loss) + 1)
m = len(model.val_loss)  # có thể nhỏ hơn vì early stopping
first_train_loss = model.train_loss[0] if len(model.train_loss) else 1.0
first_val_loss   = model.val_loss[0]   if m > 0 else 1.0
first_val_mae    = model.val_mae[0]    if m > 0 else 1.0

train_loss_rel = [z / (first_train_loss + eps) for z in model.train_loss[:m]]
val_loss_rel   = [z / (first_val_loss   + eps) for z in model.val_loss]
val_mae_rel    = [z / (first_val_mae    + eps) for z in model.val_mae]
val_r2         = model.val_r2

plt.figure(figsize=(11,6))
plt.plot(epochs_seen[:m], train_loss_rel,  marker='o', linestyle='-',  label='Train loss (rel)')
plt.plot(epochs_seen[:m], val_loss_rel,    marker='s', linestyle='--', label='Val loss (rel)')
plt.plot(epochs_seen[:m], val_r2,          marker='^', linestyle='-.', label='Val R²')
plt.plot(epochs_seen[:m], val_mae_rel,     marker='d', linestyle=':',  label='Val MAE (rel)')
plt.title('Graphic: loss (standardization) & metrics (Val)')
plt.xlabel('Epoch'); plt.ylabel('Value (standardized for loss/MAE)')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
