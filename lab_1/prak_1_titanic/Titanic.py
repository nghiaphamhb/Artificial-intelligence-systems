import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# load dataset 
titanic = pd.read_csv(r"D:\Desktop\IMPORTANT_STUDY\Системы_искусственного_интеллекта\prak_1_titanic\Titanic-Dataset.csv")

# Explore data
print("Dataset shape:", titanic.shape)
print("\nFirst 5 rows:")
print(titanic.head())
print("\nMissing values:")
print(titanic.isnull().sum())

# data preprocessing
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
titanic = titanic.copy()

titanic["Sex"] = titanic["Sex"].map({'male': 0, 'female': 1}) # Convert categorical

for ft in features: # Fill NA 
    if titanic[ft].isnull().sum() > 0:
        if titanic[ft].dtype == 'object':
            titanic[ft] = titanic[ft].fillna(titanic[ft].mode()[0])
        else:
            titanic[ft] = titanic[ft].fillna(titanic[ft].median())

X = titanic[features].values
y = titanic["Survived"].values

# split dataset into train/test and scale data
x_train_raw, x_test_raw, y_train, y_test = train_test_split(   #split
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler() # scale
x_train = scaler.fit_transform(x_train_raw)  # also fit 
x_test  = scaler.transform(x_test_raw)       

# Train model
from LogisticRegressionNP import LogisticRegressionNP

model = LogisticRegressionNP()
model.fit(x_train, y_train, lr=0.01, num_iter=1000, validation_data=(x_test, y_test), log_every=20)

# test the model  
y_pred = model.predict(x_test)

# review the model based on the test
print("=== REVIEW MODEL (Test) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")

# draw graphs
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.plot(model.loss_history, '-', linewidth=2)
plt.title('Learning Curve - Loss Reduction')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(model.metric_epochs, model.accuracy_history, 'o-', linewidth=2, label='Accuracy (Val)')
plt.plot(model.metric_epochs, model.f1_history, 'o-', linewidth=2, label='F1 (Val)')
plt.title('Metrics on Validation')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
