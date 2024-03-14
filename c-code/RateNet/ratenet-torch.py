import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# 导入所需的PyTorch模块
import torch.nn.functional as F
import torchvision.transforms as transforms
import time

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, x_data, y_label):
        self.x_data = x_data
        self.y_label = y_label

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_label[idx]

# 加载 pickle 文件
def load_pickle(pickle_path):
    try:
        with open(pickle_path, 'rb') as file:
            data = pickle.load(file)
        return np.array(data)[0, :]

    except FileNotFoundError:
        print(f"错误: 文件 '{pickle_path}' 未找到.")
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")

# 加载 Excel 文件
def load_datamark(xlsx_path):
    try:
        df = pd.read_excel(xlsx_path)
        data = df.iloc[:, 0:5].values.tolist()
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# 计算评估指标
def metrics(y_fit, y_act):
    evs = explained_variance_score(y_act, y_fit)
    mae = mean_absolute_error(y_act, y_fit)
    mse = mean_squared_error(y_act, y_fit)
    r2 = r2_score(y_act, y_fit)
    return mae, mse, evs, r2

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self, input_data_shape, regress=True):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=2, out_channels=8, kernel_size=(3, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3))
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3))
        self.batch_norm = nn.BatchNorm3d(64)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 2 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        if regress:
            self.output_layer = nn.Linear(128, 1)
        else:
            self.output_layer = nn.Linear(128, 2)

        self.regress = regress

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.batch_norm(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.regress:
            x = self.output_layer(x)
        else:
            x = F.softmax(self.output_layer(x), dim=1)
        return x

#%%
if __name__ == "__main__":
    x_data = []
    y_label = []
    start_time = time.time()# 得到运行开始时间
    #xlsx_path='/Users/liz/Documents/Repos/graduate-dissertation/f-file/datamark.xlsx'
    xlsx_path=r'C:\Users\lizyu\Desktop\graduate-dissertation\f-file\datamark.xlsx'
    training_component=load_datamark(xlsx_path)
    for lines in training_component:
        if lines[4]==1:
            pickle_path=r'D:/RateNet/pickle/' + lines[0] + '-' + lines[1] + '-' + str(lines[2]) + '.pkl'
            print(pickle_path)
    
    
            training_data_for_instance = load_pickle(pickle_path)
    
    
            print("shape of this pickle is ",np.shape(training_data_for_instance))
    
    
            for i in list(range(0,np.shape(training_data_for_instance)[0],100)):
                x_data.append(np.average(training_data_for_instance[i:i+100,:,:,:,:],axis=0))
                y_label.append(lines[3])

    print(y_label)
        
    x = np.transpose(x_data,(0,4,1,2,3))
    print(np.shape(x))
    y = y_label
    print(np.shape(y))
    #%%
    # 分割数据
    x_train = np.empty((0, 2, 20, 20, 20))
    y_train = []
    x_test = np.empty((0, 2, 20, 20, 20))
    y_test = []

    print(np.shape(x_train), np.shape(y_train))

    for i in [0, 1, 2,3]:
        x_train = np.concatenate((x_train, x[i::5, :, :, :, :]), axis=0)
        y_train = np.concatenate((y_train, y[i::5]))
    for i in [4]:
        x_test = np.concatenate((x_test, x[i::5, :, :, :, :]), axis=0)
        y_test = np.concatenate((y_test, y[i::5]))

    print(np.shape(x_train), np.shape(y_train))
    #interpolation
    x_train_xy = np.rot90(x_train, k=1, axes=(3,4))
    x_train_xz = np.rot90(x_train, k=1, axes=(2, 3))
    x_train_yz = np.rot90(x_train, k=1, axes=(2, 4))
    x_train = np.concatenate((x_train, x_train_xy, x_train_xz, x_train_yz), axis=0)
    print(np.shape(x_train))
    y_train = np.concatenate((y_train, y_train, y_train, y_train))
    print(np.shape(y_train))

    device = torch.device("cuda")

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    #%%
    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)

    model = CNN(input_data_shape=(2, 20, 20, 20)).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    num_epochs = 500
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

    print("训练完毕")
    end_time = time.time()
    execution_time = end_time - start_time

    print("导入和训练时间为：", execution_time, "秒")
    # 保存模型
    torch.save(model.state_dict(), r'D:\RateNet\ratenet.pt')
    #%%
    model = CNN(input_data_shape=(2, 20, 20, 20)).to(device)

# 加载预训练的模型参数
    model.load_state_dict(torch.load(r'D:\RateNet\ratenet.pt'))
    model.eval()
    with torch.no_grad():
        # 将测试集转换为PyTorch张量
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # 进行预测
        outputs = model(x_test)
        y_pred = outputs.squeeze().cpu().numpy()
        y_true = y_test.cpu().numpy()

        # 计算评估指标
        mae, mse, evs, r2 = metrics(y_pred, y_true)
        print("MAE:", mae)
        print("MSE:", mse)
        print("EVS:", evs)
        print("R2:", r2)

# %%
