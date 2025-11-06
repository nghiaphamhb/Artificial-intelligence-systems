import numpy as np

class LogisticRegressionNP:
    def __init__(self):
        self.W = None
        self.b = None
        self.loss_history = []
        self.accuracy_history = []
        self.f1_history = []
        self.metric_epochs = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def compute_loss(self, y_true, y_pred_proba):
        n = len(y_true)
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1 - eps)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)) / n
    
    def accuracy_score_np(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def f1_score_np(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-15)
        recall    = tp / (tp + fn + 1e-15)
        return 2 * precision * recall / (precision + recall + 1e-15)

    def fit(self, X, y, lr=0.01, num_iter=1000, validation_data=None, log_every=100):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0

        if validation_data:
            X_val, y_val = validation_data
        else:
            X_val, y_val = X, y

        for i in range(num_iter):
            linear = np.dot(X, self.W) + self.b
            y_pred_proba = self.sigmoid(linear)

            # loss
            loss = self.compute_loss(y, y_pred_proba)
            self.loss_history.append(loss)

            # gradient descent
            dW = (1 / n_samples) * np.dot(X.T, (y_pred_proba - y))
            db = (1 / n_samples) * np.sum(y_pred_proba - y)

            self.W -= lr * dW
            self.b -= lr * db

            # log metrics
            if i % log_every == 0:
                y_pred_val = self.predict(X_val)
                acc = self.accuracy_score_np(y_val, y_pred_val)
                f1 = self.f1_score_np(y_val, y_pred_val)
                self.accuracy_history.append(acc)
                self.f1_history.append(f1)
                self.metric_epochs.append(i)
                print(f"Epoch {i}: loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}")

    def predict_probability(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X, threshold=0.5):
        probs = self.predict_probability(X)
        return (probs > threshold).astype(int)
    
    


