import numpy as np

# Hàm này đưa các trọng số nhỏ hơn tau về 0 và giữ lại các trọng số lớn hơn.
def _soft_threshold(x, tau):
    # prox for L1: sign(x) * max(|x|-tau, 0)
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

class LinearRegressionNP2:
    """
    penalty: 'none' | 'l2' | 'l1' | 'elasticnet'
    lamda  : độ mạnh regularization (λ)
    alpha: chỉ dùng cho elasticnet (0..1). 1 -> L1 thuần, 0 -> L2 thuần

    Ghi log mỗi epoch:
      - train_loss (MSE + penalty), val_loss
      - train_mae, val_mae
      - train_r2,  val_r2
    Early stopping theo val_loss.
    """
    def __init__(self, penalty='none', lamda=0.0, alpha=0.5):
        self.penalty   = penalty
        self.lamda     = float(lamda)
        self.alpha  = float(alpha)
        self.weights   = None  # w[0] = bias
        # histories
        self.train_loss, self.val_loss = [], []
        self.train_mae,  self.val_mae  = [], []
        self.train_r2,   self.val_r2   = [], []

    # ----- metrics -----
    def compute_metrics(self, y_true, y_pred):
        mse  = np.mean((y_pred - y_true) ** 2)
        mae  = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(mse)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-15
        r2 = 1.0 - ss_res / ss_tot
        return mse, mae, rmse, r2

    # ----- penalty value trên w[1:] (không phạt bias w[0]) -----
    def _penalty_value(self, w):
        if self.lamda <= 0 or self.penalty == 'none':
            return 0.0
        v = w[1:]
        if self.penalty == 'l2':
            return self.lamda * 0.5 * np.sum(v * v)
        if self.penalty == 'l1':
            return self.lamda * np.sum(np.abs(v))
        if self.penalty == 'elasticnet':
            l1 = self.alpha * np.sum(np.abs(v))
            l2 = (1 - self.alpha) * 0.5 * np.sum(v * v)
            return self.lamda * (l1 + l2)
        return 0.0

    # Hàm huấn luyện mô hình hồi quy tuyến tính với các tùy chọn regularization khác nhau.
    def fit(self, X, y, learning_rate=0.01, epochs=1000,
            validation_data=None, patience=50, tol=1e-6, verbose=True):
        n, d = X.shape
        self.weights = np.zeros(d)

        best_val = np.inf
        best_w = self.weights.copy()
        wait = 0

        for epoch in range(epochs):
            # ---- forward ---- Tính giá trị dự đoán, tính các metrics và loss, lưu lịch sử chúng 
            y_pred = X @ self.weights
            mse, mae, _, r2 = self.compute_metrics(y, y_pred)
            loss = mse + self._penalty_value(self.weights)

            self.train_loss.append(loss)
            self.train_mae.append(mae)
            self.train_r2.append(r2)

            # ---- validation ---- Tính toán trên tập validation (nếu có), lưu lịch sử metrics và kiểm tra early stopping
            if validation_data is not None:
                Xv, yv = validation_data
                yv_pred = Xv @ self.weights
                vmse, vmae, _, vr2 = self.compute_metrics(yv, yv_pred)
                vloss = vmse + self._penalty_value(self.weights)

                self.val_loss.append(vloss)
                self.val_mae.append(vmae)
                self.val_r2.append(vr2)

                # early stopping theo val_loss
                if vloss + tol < best_val:
                    best_val = vloss
                    best_w = self.weights.copy()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"[Early stop] epoch={epoch}, best val loss={best_val:.6f}")
                        self.weights = best_w
                        break

            # === backward + update weights ===
            # ---- gradient step (data + L2 phần của ENet) ----
            grad = (2.0 / n) * (X.T @ (y_pred - y))   # dMSE/dw
            if self.lamda > 0 and self.penalty in ('l2', 'elasticnet'):
                l2_coeff = self.lamda if self.penalty == 'l2' else self.lamda * (1 - self.alpha)
                grad[1:] += l2_coeff * self.weights[1:]   # d/dw (0.5*||w||^2) = w

            w_new = self.weights - learning_rate * grad

            # ---- proximal step cho phần L1 (L1 hoặc ENet) ----
            if self.lamda > 0 and self.penalty in ('l1', 'elasticnet'):
                l1_coeff = self.lamda if self.penalty == 'l1' else self.lamda * self.alpha
                tau = learning_rate * l1_coeff
                w_new[1:] = _soft_threshold(w_new[1:], tau)

            self.weights = w_new

            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch:4d} | train_loss={loss:.6f} (mse={mse:.6f}) | R2={r2:.4f}"
                if validation_data is not None:
                    msg += f" | val_loss={vloss:.6f} | val_R2={vr2:.4f}"
                print(msg)

    def predict(self, X):
        return X @ self.weights
