import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from nn_class import *
file_path = r"C:\Users\admin\Desktop\census+income+kdd\census\census-income.data"

# 需要的列
required_columns = [
    "age", "class_worker", "education", "marital_status", "race",
    "sex", "capital_gain", "capital_loss", "income"
]

# 所有列名（示例）
all_columns = [
    "age", "class_worker", "detailed_industry_recode", "detailed_occupation_recode",
    "education", "wage_per_hour", "enroll_in_edu_inst_last_wk", "marital_status",
    "major_industry_code", "major_occupation_code", "race", "hispanic_origin", "sex",
    "member_of_a_labor_union", "reason_for_unemployment", "full_or_part_time_employment_stat",
    "capital_gain", "capital_loss", "dividends_from_stocks", "tax_filer_stat", "region_of_previous_residence",
    "state_of_previous_residence", "detailed_household_and_family_stat", "detailed_household_summary_in_household",
    "instance_weight", "migration_code_change_in_msa", "migration_code_change_in_reg", "migration_code_move_within_reg",
    "live_in_this_house_1_year_ago", "migration_prev_res_in_sunbelt", "num_persons_worked_for_employer",
    "family_members_under_18", "country_of_birth_father", "country_of_birth_mother", "country_of_birth_self",
    "citizenship", "own_business_or_self_employed", "veterans_benefits", "weeks_worked_in_year", "year",
    "income"
]


# 读取文件（去掉多余列，选择需要的列）
df = pd.read_csv(file_path, header=None, names=all_columns, skipinitialspace=True, na_values=["?"])
df = df[required_columns]  # 选择需要的列
print(set(df["income"].values))
# 查看前几行
print(df.head())

df.dropna(inplace=True)

# 目标变量处理：'income'列，转换为0和1（<=50K -> 0, >50K -> 1）
df['income'] = df['income'].apply(lambda x: 1 if x == '50000+.' else 0)

#标签编码
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# df = pd.read_csv('data.csv')
X = df.drop('income', axis=1)
y = df['income']

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=16)


# class CensusIncomeNN(nn.Module):
#     def __init__(self, input_size):
#         super(CensusIncomeNN, self).__init__()
#         # 深度神经网络，增加更多隐藏层
#         self.layer1 = nn.Linear(input_size, 128)
#         self.layer2 = nn.Linear(128, 1)
#         self.layer3 = nn.Linear(64, 1)
#         self.layer4 = nn.Linear(32, 16)
#         self.layer5 = nn.Linear(16, 8)
#         self.output = nn.Linear(8, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
#
#     def forward(self, x):
#         x = self.relu(self.layer1(x))
#         # x = self.relu(self.layer2(x))
#         # x = self.relu(self.layer3(x))
#         # x = self.relu(self.layer4(x))
#         # x = self.sigmoid(self.layer5(x))
#         x = self.sigmoid(self.layer2(x))
#         return x
#
# # 初始化模型、损失函数和优化器
# input_size = X_train.shape[1]
# model = CensusIncomeNN(input_size)
#
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# def train_model(model, X_train, y_train, epochs=2000, batch_size=1164):
#     model.train()
#     for epoch in range(epochs):
#         inputs = torch.tensor(X_train, dtype=torch.float32)
#         labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#         epoch_loss = 0.0
#         for i in range(0, len(X_train), batch_size):
#             batch_inputs = inputs[i:i + batch_size]
#             batch_labels = labels[i:i + batch_size]
#             optimizer.zero_grad()
#             outputs = model(batch_inputs)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
#
# # 评估模型
# def evaluate_model(model, X_test, y_test):
#     model.eval()
#     with torch.no_grad():
#         inputs = torch.tensor(X_test, dtype=torch.float32)
#         labels = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
#         outputs = model(inputs)
#         predictions = (outputs >= 0.5).int()
#         accuracy = (predictions.eq(labels).sum().item()) / len(y_test)
#         print(f"Test Accuracy: {accuracy:.4f}")
#
# # 保存模型
# def save_model(model, path="flag_model.pth"):
#     torch.save(model.state_dict(), path)
#     print(f"Model saved to {path}")
#
# # 加载模型
# def load_model(path, input_size):
#     # 重新定义与保存时相同架构的模型
#     loaded_model = CensusIncomeNN(input_size)
#     loaded_model.load_state_dict(torch.load(path))
#     loaded_model.eval()  # 切换到评估模式
#     print(f"Model loaded from {path}")
#     return loaded_model

k, a = 6, 0.5
input_size = X_train.shape[1]
model = CensusIncomeNN(input_size)

# 影子防御
for i in range(0, k+1):
    # 构建影子数据集
    import numpy as np

    x2_train, _, _, _ = train_test_split(X_scaled, y, test_size=0.5, random_state=82)
    x2_scaled = scaler.fit_transform(x2_train)
    inputs = torch.tensor(x2_train, dtype=torch.float32)
    # 调用加载函数
    if i == 120:
        loaded_model = load_model(f"flag_model.pth", input_size)
    else:
        loaded_model = load_model(f"./model/flag_model{i}.pth", input_size)
    outputs = loaded_model(inputs)
    predictions = (outputs >= 0.5).int()
    y_label = predictions

    noise = np.hstack((x2_train, y_label))
    original = df.values

    sampled_noise = noise[np.random.choice(noise.shape[0], int(2000000 * a)), :]

    # 从 original 中随机抽取a 条
    sampled_original = original[np.random.choice(original.shape[0], int(2000000 * (1 - a))), :]

    # 将两个数据集合并为一个新的数据集
    combined = np.vstack((sampled_noise, sampled_original))
    print(combined)
    X, y = combined[:, :-1], combined[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=12)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, X_train, y_train, epochs=200)


    evaluate_model(model, X_test, y_test)


    # 调用保存函数
    save_model(model, path=f"flag_model{i+1}.pth")
