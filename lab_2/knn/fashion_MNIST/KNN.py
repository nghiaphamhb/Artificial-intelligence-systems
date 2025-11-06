from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import numpy as np

class KNN:
    def __init__(self, k=3, metric='euclidean', p=2):
        """
        :param k: số lượng hàng xóm gần nhất
        :param metric: loại khoảng cách ('euclidean', 'manhattan', 'minkowski', 'cosine')
        :param p: tham số p cho Minkowski (chỉ dùng khi metric='minkowski')
        """
        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        # KDTree chỉ dùng cho Euclidean hoặc Manhattan
        if self.metric in ['euclidean', 'manhattan']:
            self.tree = KDTree(X_train)
        else:
            self.tree = None  # Cosine hoặc Minkowski dùng cdist

    def predict(self, X_test):
        if self.tree is not None:
            # KDTree query
            p = 2 if self.metric == 'euclidean' else 1
            dist, indices = self.tree.query(X_test, k=self.k, p=p)   #dist: khoảng cách từ mỗi điểm test đến k neighbor gần nhất; indices: vị trí (index) của k neighbor gần nhất trong X_train
        else:
            # cdist cho Minkowski hoặc Cosine
            if self.metric == 'minkowski':
                distances = cdist(X_test, self.X_train, metric='minkowski', p=self.p)
            elif self.metric == 'cosine':
                distances = cdist(X_test, self.X_train, metric='cosine')
            else:
                raise ValueError(f"Metric '{self.metric}'dont support.")
            indices = np.argsort(distances, axis=1)[:, :self.k]

        # Lấy nhãn của k hàng xóm gần nhất
        k_nearest_labels = self.y_train[indices]

        # Bỏ phiếu nhãn phổ biến nhất
        y_pred = np.array([np.bincount(labels).argmax() for labels in k_nearest_labels])
        return y_pred
