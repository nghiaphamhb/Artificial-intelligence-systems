import pandas as pd
import numpy as np
from KNN import KNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu CSV
train_data = pd.read_csv('/content/fashion-mnist_train.csv')
test_data  = pd.read_csv('/content/fashion-mnist_test.csv')
print("Data loaded successfully!")


# Chuẩn hóa dữ liệu (phần x - pixel quá to nên cần C/H)
X_train = train_data.drop(columns=['label']).values.astype('float32') / 255.0
y_train = train_data['label'].values.astype('int64')
X_test  = test_data.drop(columns=['label']).values.astype('float32') / 255.0
y_test  = test_data['label'].values.astype('int64')

# Thay NaN bằng 0 (điền bù vào các dòng có Nan)
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)


# Danh sách các metric
metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine']

for metric in metrics:
    print(f"\n==== KNN vs metric: {metric} ====")
    
    if metric == 'minkowski':
        knn = KNN(k=3, metric=metric, p=3)  # p=3 là ví dụ
    else:
        knn = KNN(k=3, metric=metric)

    # Huấn luyện mô hình
    knn.fit(X_train, y_train)

    # Dự đoán dữ liệu test
    y_pred = knn.predict(X_test)

    # Đánh giá chất lượng
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)

    # Vẽ heatmap ma trận nhầm lẫn
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {metric}')
    plt.show()
