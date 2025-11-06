class StandardScalerNP:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0) + 1e-12
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_