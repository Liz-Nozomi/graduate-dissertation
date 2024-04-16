import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# 检查CUDA是否可用
print("cudastatus:",torch.cuda.is_available())
 
# 显示当前CUDA版本
print("cudaversion:",torch.version.cuda)
def metrics(y_fit, y_act):
    evs = explained_variance_score(y_act, y_fit)
    mae = mean_absolute_error(y_act, y_fit)
    mse = mean_squared_error(y_act, y_fit)
    r2 = r2_score(y_act, y_fit)
    return mae, mse, evs, r2

device = torch.device("cpu")
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

def load_pickle(pickle_path):
    try:
        with open(pickle_path, 'rb') as file:
            data = pickle.load(file)
        return np.array(data)[0, :]

    except FileNotFoundError:
        print(f"错误: 文件 '{pickle_path}' 未找到.")
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")


model = CNN(input_data_shape=(2, 20, 20, 20)).to(device)
#test_pickle_path=r'D:\RateNet\pickle\MIm-MeSCN-300.pkl'
test_pickle_path=r'/Volumes/exfat/RateNet/pickle/MIm-MeSCN-360.pkl'
testing_data=load_pickle(test_pickle_path)

x_data = []
y_label = []
for i in list(range(0,np.shape(testing_data)[0],100)):
    x_data.append(np.average(testing_data[i:i+100,:,:,:,:],axis=0))
    y_label.append(38.119)
x = np.transpose(x_data,(0,4,1,2,3))
print(np.shape(x))
y = y_label
print(np.shape(y))

# 加载预训练的模型参数
#model.load_state_dict(torch.load(r'D:\RateNet\ratenet.pt'))
model.load_state_dict(torch.load(r'/Volumes/exfat/RateNet/ratenet.pt',map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
        # 将测试集转换为PyTorch张量
    x_test = torch.tensor(x, dtype=torch.float32).to(device)
    y_test = torch.tensor(y, dtype=torch.float32).to(device)

        # 进行预测
    outputs = model(x_test)
    y_pred = outputs.squeeze().cpu().numpy()
    y_true = y_test.cpu().numpy()
    print(y_pred)

    print('y_true',np.average(y_true))
    print('y_pred',np.average(y_pred))
    print('median',np.median(y_pred))
        # 计算评估指标
    mae, mse, evs, r2 = metrics(y_pred, y_true)
    print("MAE:", mae)
    print("MSE:", mse)
    print("EVS:", evs)
    print("R2:", r2)