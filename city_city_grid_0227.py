import torch
import pandas as pd
import os
from torch_geometric.nn import EdgeCNN
from torch.nn import CrossEntropyLoss
import numpy as np
from torch_geometric.data import Data
import re
import matplotlib.pyplot as plt
import seaborn as sns

EPOCH = 20  # 训练次数，epoch， 范围为【0，++】
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("computer device is:", device)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def get_citylatlon_data(cityname, latNum, lonNum):
    filenames = os.listdir('./Mydata0120')
    edge_index_filenames = [x1 for x1 in filenames if
                            re.match(rf"edge_index{cityname}lat{latNum}lon{lonNum}.csv", x1, re.IGNORECASE)]
    x_filenames = [x1 for x1 in filenames if re.match(rf"x{cityname}lat{latNum}lon{lonNum}.csv", x1, re.IGNORECASE)]
    city_data = []
    if not edge_index_filenames or not x_filenames:
        return []
    for file in edge_index_filenames:
        edge_index2 = pd.read_csv(os.path.join('./Mydata0120/', file)).to_numpy()
        if edge_index2.shape[1] < 10:
            continue
        edge_index1 = torch.from_numpy(edge_index2)
        for x_file in x_filenames:
            if x_file == file[len('edge_inde'):]:
                x2 = pd.read_csv(os.path.join('./Mydata0120/', x_file))
                x2 = (x2 - x2.min()) / (x2.max() - x2.min())
                x2 = x2.to_numpy()
                x2 = torch.from_numpy(x2).float()
                city_data = Data(edge_index=edge_index1, x=x2)
    return city_data


def modify_data(data: Data, bias: int):
    data_list = []
    edge_index = data.edge_index + bias
    y = data.y
    for x in data.x:
        data_list.append(Data(edge_index=edge_index, x=x.reshape(2, -1), y=y))
    return data_list


def modify_data2(data1: Data, data2: Data):
    x = torch.cat((data1.x, data2.x), 0)
    edge_index = torch.cat((data1.edge_index, data2.edge_index + data1.x.shape[0]), 1)
    y = [*[[1, 0] for i2 in range(data1.x.shape[0])], *[[0, 1] for i2 in range(data2.x.shape[0])]]
    y = torch.tensor(y).reshape(-1, 2)
    return Data(x=x, edge_index=edge_index, y=y)


def fit(epoch, data_train):
    model = EdgeCNN(in_channels=2, out_channels=2, num_layers=8, hidden_channels=128,
                    dropout=0.1).to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()  # 核心作用是在模型训练期间启用 Batch Normalization 和 Dropout 功能
    edge_index, x, y = data_train.edge_index.to(device), data_train.x.to(device), data_train.y.to(device)

    for ep in range(epoch):
        output = model(x=x, edge_index=edge_index)
        y = y.type(torch.FloatTensor)
        y = y.to(device)
        loss = loss_fn(output.reshape(1, -1), y.reshape(1, -1))
        print(f"第{ep}次训练，loss为：{loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _right = 0
    _error = 0
    for index, ot in enumerate(output):
        if ot[0] < ot[1] and y[index][0] < y[index][1]:
            _right += 1
        elif ot[0] > ot[1] and y[index][0] > y[index][1]:
            _right += 1
        else:
            _error += 1
    return _right / (_right + _error)


if __name__ == '__main__':
    pwd = './GNNTrainTest0120/'
    mkdir(pwd)
    cities = []
    f = open("./world_city_20240217.txt", "r")
    for line in f.readlines():
        linestr = line.strip()
        linestrlist = linestr.split("\t")
        cities.append(linestrlist)

    # city_list0 = ['Dubai']
    # city_list1 = ['Madrid']
    city_list0 = ['Frankfurt', 'Frankfurt', 'Los Angeles', 'Los Angeles', 'Tokyo', 'Tokyo']
    city_list1 = ['Warsaw', 'Mexico City', 'Frankfurt', 'Tokyo', 'Seoul', 'Los Angeles']
    for i in range(len(city_list0)):
        city0 = city_list0[i]
        city1 = city_list1[i]

        TestAcc = np.full((20, 20), np.nan)  # np.zeros((20, 20))
        for lat in range(20):
            for lon in range(20):
                citylatlon_0 = get_citylatlon_data(city0, latNum=lat, lonNum=lon)
                citylatlon_1 = get_citylatlon_data(city1, latNum=lat, lonNum=lon)
                if not citylatlon_1 or not citylatlon_0:
                    print(f"位置{(lat, lon)}内容找不到，自动跳过")
                    # TestAcc[lat][lon] = 0
                    continue
                citylatlonPair_data = modify_data2(citylatlon_0, citylatlon_1)

                # epoch_acc = np.nan
                epoch_acc = fit(EPOCH, citylatlonPair_data)
                TestAcc[19-lat][lon] = epoch_acc
    # TestAcc = pd.read_excel("./grid/TokyoLos Angeles.xlsx", sheet_name="Sheet1")
    # TestAcc = TestAcc.iloc[:, 1:]
        print(TestAcc)
        fig, ax = plt.subplots()
        mask = np.isnan(TestAcc)
        # sns.set_context({'figure.figsize': (8, 16)})
        # sns.heatmap(data=TestAcc, annot=True, fmt=".2f")
        sns.heatmap(data=TestAcc, cmap='coolwarm', square=True, mask=mask, alpha=0.6)
        # cmap='YlGnBu', linewidths=0.001, annot=True, fmt='.1f', linewidths=0.01, linecolor="grey"
        for i1 in range(len(TestAcc)):
            for j1 in range(len(TestAcc[i1])):
                if not np.isnan(TestAcc[i1][j1]):
                    ax.add_patch(plt.Rectangle((j1, i1), 1, 1, fill=False, edgecolor='grey', lw=0.5))
        plt.xticks([])
        plt.yticks([])
        # plt.title('')
        plt.savefig('./grid/' + city0 + city1 + '.png')
        # plt.show()
        # plt.clf()
        # TestAcc = pd.DataFrame(TestAcc)
        # TestAcc.to_excel('./grid/' + city0 + city1 + '.xlsx')
