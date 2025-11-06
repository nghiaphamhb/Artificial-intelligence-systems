import pandas as pd

# Đọc dữ liệu theo từng phần nhỏ
chunk_size = 10000  # Kích thước mỗi phần đọc
train_chunks = pd.read_csv(r'D:\Desktop\IMPORTANT_STUDY\Системы_искусственного_интеллекта\lab_2\knn\fashion_MNIST\fashion-mnist_train.csv', chunksize=chunk_size)

# Tạo dataframe từ các phần nhỏ
train_data = pd.concat(train_chunks, ignore_index=True)

# Kiểm tra dữ liệu
print(train_data.head())
