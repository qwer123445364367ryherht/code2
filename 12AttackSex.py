import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd
from nn_class import *


X, y = load_data()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=16)
# model = load_model('census_income_model.pth', X_train.shape[1])
model = load_model('./model/flag_model1.pth', X_train.shape[1])
# model = load_model('./model/l_model1.pth', X_train.shape[1])

# 定义攻击模型
class AttackModel(nn.Module):
    def __init__(self, input_size):
        super(AttackModel, self).__init__()
        # 深度神经网络DNN
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# 构建训练和攻击数据
def generate_attack_data(model, X_train, y_train, target_column_index):
    """
    利用目标模型的输出生成攻击数据。
    :param model:已训练的目标模型
    :param X_train: 原始训练数据
    :param y_train: 原始训练标签
    :param target_column_index: 被攻击的特征列索引
    :return: 攻击模型所需的输入和目标标签
    """
    model.eval()
    with torch.no_grad():
        # 去除敏感特征（如 'sex'）
        X_without_target = np.delete(X_train, target_column_index, axis=1)

        # 模型预测
        inputs = torch.tensor(X_train, dtype=torch.float32)
        model_outputs = model(inputs).numpy()

        # 攻击数据：特征为去掉目标列后的数据与预测值拼接
        attack_inputs = np.hstack((X_without_target, model_outputs))

        # 目标为敏感特征的真实值
        target_labels = X_train[:, target_column_index]

    return attack_inputs, target_labels


# 攻击模型训练
def train_attack_model(attack_model, attack_inputs, target_labels, epochs=500, batch_size=512):
    """
    训练攻击模型，用于推断敏感特征。
    :param attack_model: 攻击模型
    :param attack_inputs: 攻击数据特征
    :param target_labels: 攻击目标标签
    :param epochs: 训练轮数
    :param batch_size: 批量大小
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

    # 数据准备
    X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(
        attack_inputs, target_labels, test_size=0.2, random_state=42)

    X_attack_train = torch.tensor(X_attack_train, dtype=torch.float32)
    X_attack_test = torch.tensor(X_attack_test, dtype=torch.float32)
    y_attack_train = torch.tensor(y_attack_train, dtype=torch.float32).view(-1, 1)
    y_attack_test = torch.tensor(y_attack_test, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        attack_model.train()
        epoch_loss = 0.0
        for i in range(0, len(X_attack_train), batch_size):
            batch_inputs = X_attack_train[i:i + batch_size]
            batch_labels = y_attack_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = attack_model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 测试模型
    # 测试模型
    attack_model.eval()
    with torch.no_grad():
        test_outputs = attack_model(X_attack_test)
        test_preds = (test_outputs.numpy() > 0.5).astype(int)  # 将连续值转换为二进制值
        y_attack_test_binary = y_attack_test.numpy().astype(int)  # 确保测试标签也是整型
        test_acc = accuracy_score(y_attack_test_binary, test_preds)
        print(f"Attack Model Accuracy: {test_acc:.4f}")


# 生成攻击数据
target_column_index = 5  # 'sex特征的索引
attack_inputs, target_labels = generate_attack_data(model, X_train, y_train, target_column_index)

# 初始化攻击模型
attack_input_size = attack_inputs.shape[1]
attack_model = AttackModel(attack_input_size)

# 训练攻击模型
train_attack_model(attack_model, attack_inputs, target_labels)
