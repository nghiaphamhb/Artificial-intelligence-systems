import numpy as np

class LogisticRegressionNP:
    """
    penalty: 'none' | 'l2' | 'l1' | 'elasticnet'
    alpha:  λ  (>=0)
    l1_ratio: if elasticnet (0..1): 1 -> L1 thuần, 0 -> L2 thuần
    """

    # Hàm khởi tạo
    def __init__(self, penalty='none', alpha=0.0, l1_ratio=0.5):
        self.penalty = penalty
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.W = None
        self.b = None

        # Lịch sử đầy đủ theo từng epoch
        self.loss_history = []           # Tổng loss (data + penalty) theo epoch
        self.data_loss_history = []      # Chỉ data-loss (không tính penalty)
        # Logging theo mốc (log_every)
        self.metric_epochs = []
        self.val_loss_history = []
        self.val_data_loss_history = []
        self.accuracy_history = []
        self.f1_history = []

    # ===== Util =====
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Tính hàm loss gốc 
    def _data_loss(self, y_true, y_pred_proba):
        n = len(y_true)
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1 - eps)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)) / n

    # Tính penalty
    def _penalty_value(self):
        if self.penalty == 'none' or self.alpha <= 0:
            return 0.0
        W = self.W  # Không phạt bias
        if self.penalty == 'l2':
            return self.alpha * 0.5 * np.sum(W * W)
        elif self.penalty == 'l1':
            return self.alpha * np.sum(np.abs(W))
        elif self.penalty == 'elasticnet':
            l1 = self.l1_ratio * np.sum(np.abs(W))
            l2 = (1 - self.l1_ratio) * 0.5 * np.sum(W * W)
            return self.alpha * (l1 + l2)
        else:
            return 0.0

    # Tính gradient của penalty
    def _penalty_grad(self):
        if self.penalty == 'none' or self.alpha <= 0:
            return 0.0
        W = self.W
        if self.penalty == 'l2':
            # d/dW (0.5*||W||^2) = W
            return self.alpha * W
        elif self.penalty == 'l1':
            # Subgradient của |W| là sign(W); tại 0 ta lấy 0
            return self.alpha * np.sign(W)
        elif self.penalty == 'elasticnet':
            # alpha * ( l1_ratio * sign(W) + (1-l1_ratio) * W )
            return self.alpha * (self.l1_ratio * np.sign(W) + (1 - self.l1_ratio) * W)
        else:
            return 0.0

    # ===== Metrics =====
    def accuracy_score_np(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def f1_score_np(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp + 1e-15)
        recall    = tp / (tp + fn + 1e-15)
        return 2 * precision * recall / (precision + recall + 1e-15)

    # ===== API =====
    def fit(self, X, y, lr=0.01, num_iter=1000, validation_data=None, log_every=100):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0.0

        if validation_data is not None:
            X_val, y_val = validation_data
        else:
            X_val, y_val = X, y

        for i in range(num_iter):
            # Forward pass
            linear = np.dot(X, self.W) + self.b
            y_pred_proba = self.sigmoid(linear)

            # Losses
            data_loss = self._data_loss(y, y_pred_proba)
            penalty = self._penalty_value()
            total_loss = data_loss + penalty

            self.data_loss_history.append(data_loss)
            self.loss_history.append(total_loss)

            # Gradient (data term)
            dW = (1.0 / n_samples) * np.dot(X.T, (y_pred_proba - y))
            db = (1.0 / n_samples) * np.sum(y_pred_proba - y)

            # + Gradient của penalty (không phạt bias)
            pen_grad = self._penalty_grad()
            if isinstance(pen_grad, float):
                # Không có penalty (pen_grad = 0.0 )
                pass
            else:
                dW = dW + pen_grad

            # Update
            self.W -= lr * dW
            self.b -= lr * db

            # Logging theo mốc
            if i % log_every == 0 or i == num_iter - 1:
                y_val_proba = self.predict_probability(X_val)
                y_val_pred  = (y_val_proba > 0.5).astype(int)

                val_data_loss = self._data_loss(y_val, y_val_proba)
                val_total_loss = val_data_loss + self._penalty_value()

                acc = self.accuracy_score_np(y_val, y_val_pred)
                f1  = self.f1_score_np(y_val, y_val_pred)

                self.metric_epochs.append(i)
                self.val_data_loss_history.append(val_data_loss)
                self.val_loss_history.append(val_total_loss)
                self.accuracy_history.append(acc)
                self.f1_history.append(f1)

                print(f"Epoch {i:4d}: "
                      f"train_loss={total_loss:.4f} "
                      f"(data={data_loss:.4f}, pen={penalty:.4f}); "
                      f"val_loss={val_total_loss:.4f} "
                      f"(acc={acc:.4f}, f1={f1:.4f})")

    def predict_probability(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X, threshold=0.5):
        probs = self.predict_probability(X)
        return (probs > threshold).astype(int)
