import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


def load_data():
    # 需要的列
    required_columns = [
        "age", "class_worker", "education", "marital-status", "race",
        "sex", "capital-gain", "capital-loss", "income"
    ]

    all_columns = ['age', 'class_worker', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship',
                   'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, names=all_columns, sep=',\s', engine='python')
    df = df[required_columns]
    # 删除含有空值的行
    df.dropna(inplace=True)

    # 目标变量处理：'income'列，转换为0和1（<=50K -> 0, >50K -> 1）
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # 对类别变量进行标签编码
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 特征和目标分开
    X = df.drop('income', axis=1)
    y = df['income']
    df.to_csv('data.csv')
    return X, y


class CensusIncomeNN(nn.Module):
    def __init__(self, input_size):
        super(CensusIncomeNN, self).__init__()
        # 深度神经网络，增加更多隐藏层
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


# 训练模型
def train_model(model, X_train, y_train, epochs=20, batch_size=64, lr=0.0001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        try:
            labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        except:
            labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


# 评估模型
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        try:
            labels = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        except:
            labels = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        outputs = model(inputs)
        predictions = (outputs >= 0.5).int()
        accuracy = (predictions.eq(labels).sum().item()) / len(y_test)
        print(f"Test Accuracy: {accuracy:.4f}")


# 保存模型
def save_model(model, path="flag_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# 加载模型
def load_model(path, input_size):
    # 重新定义与保存时相同架构的模型
    loaded_model = CensusIncomeNN(input_size)
    loaded_model.load_state_dict(torch.load(path))
    loaded_model.eval()  # 切换到评估模式
    print(f"Model loaded from {path}")
    return loaded_model
