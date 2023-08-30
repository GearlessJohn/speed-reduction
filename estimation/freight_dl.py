import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")

data_eco = pd.read_excel("./data/projections Shipping Baseline.xlsx").dropna()
data_eco["Data"] = data_eco["Date"].astype("datetime64[ns]")
data_eco = data_eco.set_index("Data")

clarksons = pd.read_excel(
    "./data/Indices Clarksons.xlsx", skiprows=range(0, 5), sheet_name="Container"
)
clarksons["Date"] = clarksons["Date"].astype("datetime64[ns]")
clarksons = clarksons.set_index("Date")
clarksons = clarksons.resample("Q").mean()

data = pd.merge(clarksons, data_eco, left_index=True, right_index=True, how="left")
data.rename(columns={"$/day": "Clarksons"}, inplace=True)
y = data.pop("Clarksons")
X = data.drop(["Date"], axis=1)
X = X.drop(X.filter(regex='Bulker|Tanker').columns, axis=1)


# standardise the columns of numeric values
def standardise(data, list_column):
    for name_column in list_column:
        # standardize
        if data[name_column].std() != 0:
            data[name_column] = (data[name_column] - data[name_column].mean()) / data[name_column].std()
        else:
            data[name_column] = (data[name_column] - data[name_column].mean())


standardise(X, X.columns.to_list())
X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.3)
X_train = X.loc[X_train]
X_test = X.loc[X_test]

X_train_tensor = torch.Tensor(X_train.values)
X_test_tensor = torch.Tensor(X_test.values)
y_train_tensor = torch.Tensor(y_train.values).reshape((-1, 1))
y_test_tensor = torch.Tensor(y_test.values).reshape((-1, 1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, input_size, n_feature=64, output_size=1):
        super(CNN, self).__init__()
        # insert your code here
        K = 5  # Kernel size
        self.n_feature = n_feature
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=K, padding=2)
        # self.conv2 = nn.Conv1d(n_feature, n_feature, kernel_size=K, padding=2)
        self.fc1 = nn.Linear(self.n_feature * 12, output_size)
        self.dropout = nn.Dropout(p=0.5)
        # self.batchnorm = nn.BatchNorm1d(output_size, momentum=0.1)

    def forward(self, x, verbose=False):
        # initial dimensions for x will be [batch, 10]
        # insert your code here
        x = x.unsqueeze(dim=1)  # [batch, 1, 51]
        print("x.shape", x.shape)
        x = self.conv1(x)  # [batch, 5, 51]
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4)  # [batch, 5, 12]
        # x = self.conv2(x)  # [batch, 32, 16]
        x = F.relu(x)
        # x = F.max_pool1d(x, kernel_size=4)  # [batch, 32, 4]
        print("x.shape", x.shape)
        x = x.reshape(-1, self.n_feature * 12)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x


def initilize_model_CNN(input_size,
                        n_feature=64,
                        output_size=1,
                        learning_rate=0.01):
    cnn_model = CNN(input_size=input_size,
                    n_feature=n_feature,
                    output_size=output_size)

    cnn_model.to(device)

    optimizer = optim.Adam(cnn_model.parameters(),
                           lr=learning_rate)

    # L1loss is the MAE loss
    loss_function = nn.L1Loss()

    return cnn_model, optimizer, loss_function


def trainer(data_train, data_test, model, loss_fn, epoch=100, batch_size=16, rate=1e-3, train_record=[], eval_record=[],
            eval_interval=10):
    optimiser = torch.optim.Adam(model.parameters(), lr=rate)
    loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)
    for i in range(epoch):
        loss_epoch = 0
        loss_total = 0
        print("---------------------- now training epoch " + str(i) + " ----------------------")
        for step, (batch_x, batch_y) in enumerate(loader_train):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)

            loss = loss_fn(pred, batch_y)

            loss_total += loss
            train_loss = float(loss_total / (step + 1))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        train_record.append(train_loss)
        print("training epoch " + str(i) + " loss " + str(train_loss))

        if i % eval_interval == 0:
            loss_total_test = 0
            test_loss = 0
            print("---------------------- epoch " + str(i) + " now evaluating ------------------------")
            for step, (batch_x, batch_y) in enumerate(loader_test):
                # print(batch_x.float())
                batch_x = batch_x.to(device)

                batch_y = batch_y.to(device)

                pred = model(batch_x)
                # class_pred = torch.argmax(pred,dim = -1)
                loss = loss_fn(pred, batch_y)
                loss_total_test += loss
                test_loss = float(loss_total_test / (step + 1))
            # 应该要把预测结果都变成int的，但这样可能产生预测结果全是0，求不了梯度不能反向传播了，要想个办法
            eval_record.append(test_loss)

            print("evaluation epoch " + str(i) + " loss " + str(test_loss))


batch_size = 32

dataset_train = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
dataset_test = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))

cnn_model = initilize_model_CNN(input_size=X_train_tensor.shape[1], n_feature=50)

train_record = []
eval_record = []
trainer(dataset_train, dataset_test, cnn_model[0], nn.L1Loss(), epoch=200, batch_size=batch_size, rate=5 * 1e-3,
        train_record=train_record, eval_record=eval_record, eval_interval=5)

plt.plot(eval_record)
plt.show()
