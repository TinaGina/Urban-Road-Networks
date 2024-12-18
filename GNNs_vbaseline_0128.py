import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCN, GAT, GraphSAGE, GIN, FiLMConv, EdgeCNN
from torch.nn import BatchNorm1d
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
import csv

CLASS = 30  # 共有几类结果，范围为【1，30】
EPOCH = 20  # 训练次数，epoch， 范围为【0，++】
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("computer device is:", device)


class FiLM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.0):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(FiLMConv(hidden_channels, hidden_channels))
        self.convs.append(FiLMConv(hidden_channels, out_channels, act=None))

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def save_csv(list1, csv_filename):
    with open(csv_filename + 'TestAcc.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list1)
    csvfile.close()


def get_city_dataset(num, cityname):
    filenames = os.listdir('./Mydata0120')
    edge_index_filenames = [x1 for x1 in filenames if x1[:len('edge_index' + cityname)] == 'edge_index' + cityname]
    x_filenames = [x1 for x1 in filenames if x1[:len('x' + cityname)] == 'x' + cityname]
    city_dataset1 = []
    for file in edge_index_filenames:
        edge_index2 = pd.read_csv(os.path.join('./Mydata0120/', file)).to_numpy()
        edge_index1 = torch.from_numpy(edge_index2)
        for x_file in x_filenames:
            if x_file == file[len('edge_inde'):]:
                x2 = pd.read_csv(os.path.join('./Mydata0120/', x_file))
                x2 = (x2 - x2.min()) / (x2.max() - x2.min())
                x2 = x2.to_numpy()
                x2 = torch.from_numpy(x2).float()
                # # 仅取维度作为节点特征
                # x2 = pd.read_csv(os.path.join('./Mydata0120/', x_file)).to_numpy()
                # x2 = x2[:, 1]
                # x2 = torch.from_numpy(x2).float()
                # x2 = x2.reshape(-1, 1)
                y2 = torch.from_numpy(np.array(int(num))).reshape(1)
                city_data = Data(edge_index=edge_index1, x=x2, y=y2)
                city_dataset1.append(city_data)
    return city_dataset1


def fit(epoch, model, data_train, data_test):
    loss_fn = CrossEntropyLoss()
    model.train()  # 核心作用是在模型训练期间启用 Batch Normalization 和 Dropout 功能
    for index, value in enumerate(data_train):
        edge_index, x, y = value.edge_index.to(device), value.x.to(device), value.y.to(device)
        if edge_index.shape[1] < 4:
            continue
        output = model(x=x, edge_index=edge_index)
        y_pred = torch.mean(output, dim=0, keepdim=True)
        y_pred = y_pred.to(device)
        y = y.type(torch.LongTensor)
        y = y.to(device)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_correct = 0
    test_total = 0
    model.eval()  # 核心作用是在模型训练期间不启用 Batch Normalization 和 Dropout 功能
    with torch.no_grad():
        for index, value in enumerate(data_test):
            edge_index, x, y = value.edge_index.to(device), value.x.to(device), value.y.to(device)
            if edge_index.shape[1] < 4:
                continue
            output = model(x=x, edge_index=edge_index)
            y_pred = torch.mean(output, dim=0, keepdim=True)
            y_pred = y_pred.to(device)
            y = y.type(torch.LongTensor)
            y = y.to(device)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
    test_acc = test_correct / test_total
    # print(f'epoch: {epoch}, train_acc: {round(train_acc, 3)}, test_acc: {round(test_acc, 3)}')
    return test_acc


if __name__ == '__main__':
    # ================================ data process =====================================
    pwd = './GNNTrainTest0120/'
    mkdir(pwd)
    cities = []
    f = open("./world_city_20231127.txt", "r")
    for line in f.readlines():
        linestr = line.strip()
        linestrlist = linestr.split("\t")
        cities.append(linestrlist)

    # ================================ data process =====================================
    # ================================ 6 GNN models =====================================
    num_layers = 3
    hidden_channels = 128
    models = [GCN(in_channels=2, out_channels=CLASS, num_layers=num_layers, hidden_channels=hidden_channels,
                  dropout=0.1).to(device),
              GAT(in_channels=2, out_channels=CLASS, num_layers=num_layers, hidden_channels=hidden_channels, v2=True,
                  dropout=0.1).to(device)]
              # GraphSAGE(in_channels=2, out_channels=CLASS, num_layers=num_layers, hidden_channels=hidden_channels,
              #           dropout=0.1).to(device),
              # GIN(in_channels=2, out_channels=CLASS, num_layers=num_layers, hidden_channels=hidden_channels,
              #     dropout=0.1).to(device),
              # EdgeCNN(in_channels=2, out_channels=CLASS, num_layers=num_layers, hidden_channels=hidden_channels,
              #         dropout=0.1).to(device),
              # FiLM(in_channels=2, out_channels=CLASS, num_layers=num_layers, hidden_channels=hidden_channels,
              #      dropout=0.1).to(device)]
    names = ['GCN', 'GAT']  # , 'GraphSAGE', 'GIN', 'EdgeCNN', 'FiLM']
    # ================================ 6 GNN models =====================================
    # ============================ models train and test=================================
    for city_i in range(CLASS-1):  # 设置基类
        city_i_name = cities[city_i][1]
        city_dataset = get_city_dataset(city_i, cities[city_i][1])
        for city_j in range(city_i + 1, CLASS):  # 从基类以后进行二分类
            city_j_name = cities[city_j][1]
            city_dataset.extend(get_city_dataset(city_j, cities[city_j][1]))
            print('=====================================================')
            print(city_i, cities[city_i][1], 'vs', city_j, cities[city_j][1])
            print(len(city_dataset))
            labels = [int(city.y) for city in city_dataset]
            data_train, data_test, labels_train, labels_test = train_test_split(city_dataset, labels,
                                                                                test_size=0.2, random_state=42)
            for i, model in enumerate(models):
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
                test_acc = []
                for epoch in range(EPOCH):
                    epoch_test_acc = fit(epoch, model, data_train, data_test)
                    test_acc.append(epoch_test_acc)
                save_csv(test_acc, pwd + names[i])
            city_dataset = get_city_dataset(city_i, cities[city_i][1])
    print('==================================================')
