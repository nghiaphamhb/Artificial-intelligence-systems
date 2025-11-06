import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from LogisticRegressionNP import LogisticRegressionNP

# ===== 1) Load & tiền xử lý =====
titanic = pd.read_csv(r"D:\Desktop\IMPORTANT_STUDY\Системы_искусственного_интеллекта\lab_2\prak_1_titanic_advanced\Titanic-Dataset.csv")

# Chọn các đặc trưng
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

# Tiền xử lý dữ liệu: chuyển đổi giới tính, điền giá trị thiếu
titanic = titanic.copy()
titanic["Sex"] = titanic["Sex"].map({'male': 0, 'female': 1})

for ft in features:
    if titanic[ft].isnull().sum() > 0:
        if titanic[ft].dtype == 'object':
            titanic[ft] = titanic[ft].fillna(titanic[ft].mode()[0])
        else:
            titanic[ft] = titanic[ft].fillna(titanic[ft].median())

X = titanic[features].values
y = titanic["Survived"].values

# Chia tập train/test
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_raw)
x_test  = scaler.transform(x_test_raw)

# ===== 2) Huấn luyện với Logistic Regression =====
model = LogisticRegressionNP(penalty='l2', alpha=0.05)  # L2 (ridge)
# model = LogisticRegressionNP(penalty='l1', alpha=0.0001)  # L1
model = LogisticRegressionNP(penalty='elasticnet', alpha=0.01, l1_ratio=0.2)  # ElasticNet

model.fit(x_train, y_train, lr=0.05, num_iter=600, validation_data=(x_test, y_test), log_every=2)

# ===== 3) Đánh giá trên test =====
y_pred = model.predict(x_test)
print("=== REVIEW MODEL (Test) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")

# ===== 4) VẼ BIỂU ĐỒ CÓ 4 ĐƯỜNG =====
# Để 4 đường lên cùng 1 biểu đồ:
# - Train Loss (data-only) - quy về tương đối so với epoch đầu
# - Val Loss (data-only)   - quy về tương đối so với mốc log đầu
# - Val Accuracy
# - Val F1

# Đảm bảo rằng các chỉ số trong epochs là hợp lệ
epochs = model.metric_epochs

# Lấy chỉ số hợp lệ không vượt quá độ dài của val_data_loss_history
epochs = [e for e in epochs if e < len(model.val_data_loss_history)] 

# Lấy các phần tử tương ứng trong các danh sách
train_data_loss_at_logs = [model.data_loss_history[e] for e in epochs]
val_data_loss_at_logs = [model.val_data_loss_history[e] for e in epochs]

# Lấy các giá trị cho accuracy và F1
accs = [model.accuracy_history[e] for e in epochs]
f1s = [model.f1_history[e] for e in epochs]

# Vẽ đồ thị
plt.figure(figsize=(10,6))
plt.plot(epochs, train_data_loss_at_logs, linestyle='-', marker='o', label='Train loss')
plt.plot(epochs, val_data_loss_at_logs, linestyle='--', marker='s', label='Val loss')
plt.plot(epochs, accs, linestyle='-.', marker='^', label='Val Accuracy')
plt.plot(epochs, f1s, linestyle=':', marker='d', label='Val F1')

plt.title('Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy / F1')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

