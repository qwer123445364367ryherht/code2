import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from nn_class import *

X, y = load_data()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=16)

# 初始化模型、损失函数和优化器
input_size = X_train.shape[1]
model = CensusIncomeNN(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, X_train, y_train, epochs=200, lr=0.001, batch_size=512)

evaluate_model(model, X_test, y_test)

save_model(model, path="./model/flag_model1.pth")

