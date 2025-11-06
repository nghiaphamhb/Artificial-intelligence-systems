# knn_us_health_full.py
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

# ===============================
# 1. Class KNN Regression
# ===============================
class KNNRegression:
    def __init__(self, k=3, metric='euclidean', p=2):
        """
        KNN Regression với 4 loại khoảng cách: euclidean, manhattan, minkowski, cosine
        :param k: số lượng hàng xóm gần nhất
        :param metric: loại khoảng cách
        :param p: tham số p cho Minkowski
        """
        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train.astype(float)
        self.y_train = y_train.astype(float)
        # KDTree chỉ dùng Euclidean hoặc Manhattan
        if self.metric in ['euclidean', 'manhattan']:
            self.tree = KDTree(self.X_train)
        else:
            self.tree = None  # Cosine hoặc Minkowski dùng cdist

    def predict(self, X_test):
        X_test = X_test.astype(float)
        if self.tree is not None:
            p = 2 if self.metric == 'euclidean' else 1
            dist, indices = self.tree.query(X_test, k=self.k, p=p)
            if self.k == 1:
                indices = indices[:, np.newaxis]
        else:
            if self.metric == 'minkowski':
                distances = cdist(X_test, self.X_train, metric='minkowski', p=self.p)
            elif self.metric == 'cosine':
                distances = cdist(X_test, self.X_train, metric='cosine')
            else:
                raise ValueError(f"Metric '{self.metric}' dont support.")
            indices = np.argsort(distances, axis=1)[:, :self.k]
            if self.k == 1:
                indices = indices[:, np.newaxis]

        k_nearest_values = self.y_train[indices]
        y_pred = np.mean(k_nearest_values, axis=1)
        return y_pred

# ===============================
# 2. Load dataset US Health Insurance
# ===============================
data = pd.read_csv("/content/insurance.csv")
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
target = 'charges'

X = data[features].copy()
y = data[target].values

# Encode categorical
X['sex'] = X['sex'].map({'male':0, 'female':1})
X['smoker'] = X['smoker'].map({'no':0, 'yes':1})
X = pd.get_dummies(X, columns=['region'], drop_first=True)

# Chuẩn hóa numeric
numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_values = X.values.astype(float)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_values, y, test_size=0.2, random_state=42
)

# ===============================
# 3. Các metric muốn khảo sát
# ===============================
metrics = [
    ('euclidean', None),
    ('manhattan', None),
    ('minkowski', 3),  # p=3
    ('cosine', None)
]

for metric, p_val in metrics:
    print(f"\n=== Metric: {metric} ===")
    knn = KNNRegression(k=5, metric=metric, p=p_val if p_val else 2)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Thông số đánh giá
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Median AE: {median_ae:.2f}")
    print(f"R^2: {r2:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")

    # Khảo sát MSE theo k
    print("\nMSE with k (1–15):")
    for k_val in range(1, 16):
        knn_k = KNNRegression(k=k_val, metric=metric, p=p_val if p_val else 2)
        knn_k.fit(X_train, y_train)
        y_pred_k = knn_k.predict(X_test)
        mse_k = mean_squared_error(y_test, y_pred_k)
        print(f"k={k_val}: MSE={mse_k:.2f}")
