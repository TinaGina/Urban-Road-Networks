import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
import os
from torch_geometric.nn import WLConv
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
import numpy as np
from torch_geometric.data import Batch
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=ConvergenceWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
torch.manual_seed(42)

CLASS = 10


class WL(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])

    def forward(self, x, edge_index, batch=None):
        hists = []
        for conv in self.convs:
            x = conv(x, edge_index)
            hists.append(conv.histogram(x, batch, norm=True))
        return hists


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def get_city_dataset(num, cityname):
    filenames = os.listdir('./Mydata0120')
    edge_index_filenames = [x1 for x1 in filenames if x1[:len('edge_index' + cityname)] == 'edge_index' + cityname]
    city_dataset1 = []
    for file in edge_index_filenames:
        edge_index2 = pd.read_csv(os.path.join('./Mydata0120/', file)).to_numpy()
        if edge_index2.shape[1] < 10:
            continue
        edge_index1 = torch.from_numpy(edge_index2)
        x2 = torch.tensor([1 for i1 in range(edge_index1.max() + 1)]).reshape(-1, 1)
        y2 = torch.from_numpy(np.array(int(num))).reshape(1)
        city_data = Data(x=x2, y=y2, edge_index=edge_index1)
        # print(f'city_data: {city_data}')
        # print(f'x:  {city_data.x}')
        # print(f'Edge:  {city_data.edge_index.t()}')
        city_dataset1.append(city_data)
    return city_dataset1


if __name__ == '__main__':
    pwd = './GraphKernels/'
    mkdir(pwd)
    cities = []
    f = open("./world_city_20231127.txt", "r")
    for line in f.readlines():
        linestr = line.strip()
        linestrlist = linestr.split("\t")
        cities.append(linestrlist)

    city_dataset = []
    for i in range(0, CLASS):  # 修改为 len(cities)
        print('===================')
        print(i, cities[i][1])
        city_dataset.extend(get_city_dataset(cities[i][0], cities[i][1]))
        # print(city_dataset)

    data = Batch.from_data_list(city_dataset)
    print('process data success')
    # ================================================================================
    # print(f'city_dataset[0]: {city_dataset[0]}')

    wl = WL(num_layers=5)
    hists = wl(data.x, data.edge_index, data.batch)

    test_accs = torch.empty(args.runs, dtype=torch.float)
    train_headmap = [[0 for i in range(CLASS)] for j in range(CLASS)]
    test_headmap = [[0 for i in range(CLASS)] for j in range(CLASS)]

    for run in range(1, args.runs + 1):
        perm = torch.randperm(data.num_graphs)
        val_index = perm[:data.num_graphs // 10]
        test_index = perm[data.num_graphs // 10:data.num_graphs // 5]
        train_index = perm[data.num_graphs // 5:]

        best_val_acc = 0

        for _index, hist in enumerate(hists):
            train_hist, train_y = hist[train_index], data.y[train_index]
            val_hist, val_y = hist[val_index], data.y[val_index]
            test_hist, test_y = hist[test_index], data.y[test_index]
            train_y_num = torch.unique(train_y, return_counts=True)[1]
            test_y_num = torch.unique(test_y, return_counts=True)[1]

            for C in [10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]:
                model = LinearSVC(C=C, tol=0.01, dual=True)
                model.fit(train_hist, train_y)
                val_acc = accuracy_score(val_y, model.predict(val_hist))
                if val_acc > best_val_acc:
                    train_headmap = [[0 for i in range(CLASS)] for j in range(CLASS)]
                    test_headmap = [[0 for i in range(CLASS)] for j in range(CLASS)]

                    best_val_acc = val_acc
                    test_acc = accuracy_score(test_y, model.predict(test_hist))
                    test_accs[run - 1] = test_acc
                    y_pred = model.predict(train_hist)
                    for index, pre in enumerate(train_y):
                        train_headmap[pre - 1][y_pred[index] - 1] += 1

                    y_test = model.predict(test_hist)
                    for index, pre in enumerate(test_y):
                        test_headmap[pre - 1][y_test[index] - 1] += 1

                    train_headmap = np.mat(train_headmap).T/train_y_num.tolist()
                    test_headmap = np.mat(test_headmap).T/test_y_num.tolist()
                    train_headmap = train_headmap.T
                    test_headmap = test_headmap.T

        plt.figure(figsize=(9, 9))

        sns.heatmap(data=train_headmap, square=True, xticklabels=[cities[i][1] for i in range(CLASS)],
                    yticklabels=[cities[i][1] for i in range(CLASS)])
        plt.savefig(pwd + str(run) + 'train_heatmap.png')
        plt.close()

        print(f'Run: {run:02d}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

    print(f'Final Test Performance: {test_accs.mean():.4f}±{test_accs.std():.4f}')
    plt.figure(figsize=(9, 9))
    sns.heatmap(data=test_headmap, square=True, xticklabels=[cities[i][1] for i in range(CLASS)],
                yticklabels=[cities[i][1] for i in range(CLASS)])
    plt.savefig(pwd + 'test_heatmap.png')
    plt.close()
