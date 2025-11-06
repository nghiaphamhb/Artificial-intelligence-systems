import numpy as np

class LinearRegressionNP2:
    def __init__(self, l2=0.0):
        self.weights = None
        self.l2 = l2
        self.train_mse = []
        self.val_mse   = []
        self.train_r2  = []
        self.val_r2    = []

    def compute_metrics(self, y_true, y_pred):
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(mse)
        # R2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-15
        r2 = 1.0 - ss_res / ss_tot
        return mse, mae, rmse, r2

    def fit(self, X, y, learning_rate=0.01, epochs=1000, validation_data=None,
            patience=50, tol=1e-6, verbose=True):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        best_val = np.inf
        wait = 0
        best_w = self.weights.copy()

        for epoch in range(epochs):
            # forward
            y_pred = X @ self.weights
            # train metrics
            mse, _, _, r2 = self.compute_metrics(y, y_pred)
            # L2 penalty (không phạt bias w[0])
            reg = self.l2 * np.sum(self.weights[1:]**2)
            mse_reg = mse + reg
            self.train_mse.append(mse_reg)
            self.train_r2.append(r2)

            # validation
            vmse_reg = None
            vr2 = None
            if validation_data is not None:
                Xv, yv = validation_data
                yv_pred = Xv @ self.weights
                vmse, _, _, vr2 = self.compute_metrics(yv, yv_pred)
                vmse_reg = vmse + self.l2 * np.sum(self.weights[1:]**2)
                self.val_mse.append(vmse_reg)
                self.val_r2.append(vr2)

                # early stopping
                if vmse_reg + tol < best_val:
                    best_val = vmse_reg
                    best_w = self.weights.copy()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"[Early stop] epoch={epoch}, best val MSE={best_val:.6f}")
                        self.weights = best_w
                        break

            # gradient (with L2, no penalty on bias)
            grad = (2/n_samples) * (X.T @ (y_pred - y))
            grad[1:] += 2 * self.l2 * self.weights[1:]
            self.weights -= learning_rate * grad

            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch:4d} | train MSE={mse:.6f}, R2={r2:.4f}"
                if validation_data is not None:
                    msg += f" | val MSE={vmse:.6f}, val R2={vr2:.4f}"
                print(msg)

    def predict(self, X):
        return X @ self.weights