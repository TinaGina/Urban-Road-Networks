import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import os
from torch_geometric.nn import WLConv
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
from torch_geometric.data import Batch

warnings.filterwarnings('ignore', category=ConvergenceWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
torch.manual_seed(42)

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
# dataset = TUDataset(path, name='ENZYMES')
# data = Batch.from_data_list(dataset)


class MyOwnDataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        super().__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置,这个例子中是随机出创建的，所以这个文件夹为空
        return ['origin_dataset']

    @property
    def processed_file_names(self):
        return ['city_dataset.pt']

    def download(self):  # 这个例子中不是从网上下载的，所以这个函数pass掉
        pass

    def process(self):
        # 处理数据的函数,最关键（怎么创建，怎么保存）
        # 创建了100个样本，每个样本是一个图，每个图有32个节点，每个节点3个特征，每个图有42个边
        data_list = [city_data for _ in range(2)]
        data_save, data_slices = self.collate(data_list)
        # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        torch.save((data_save, data_slices), self.processed_file_names[0])


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def get_adj_degree(cityname):
    filenames = os.listdir('./Mydata')
    filenames = [x1 for x1 in filenames if x1[:len(cityname)] == cityname]
    adj_all1 = []
    degree_all1 = []
    y_all1 = []
    for file in filenames:
        adj1 = pd.read_csv(os.path.join('./Mydata', file)).to_numpy()
        adj1 = torch.from_numpy(adj1)
        edge_index = torch.nonzero(adj1).transpose(0, 1)  # .reshape(2, -1)
        degree = torch.sum(adj1.float(), dim=0).reshape(-1, 1)
        adj_all1.append(edge_index)
        degree_all1.append(degree)
    return adj_all1, degree_all1


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


if __name__ == '__main__':
    pwd = './ClusterImage/'
    mkdir(pwd)
    cities = []
    f = open("./world_city_20231127.txt", "r")
    for line in f.readlines():
        linestr = line.strip()
        linestrlist = linestr.split("\t")
        cities.append(linestrlist)
    adj_all = []
    degree_all = []
    y_all = []
    print(cities[0][0], cities[0][1], len(adj_all), len(degree_all))
    for i in range(0, len(cities)):    # 修改为 len(cities)
        print('===================')
        adj0, degree0 = get_adj_degree(cities[i][1])
        y0 = np.arange(len(degree0)) * i
        print(cities[i][0], cities[i][1], len(adj0), len(degree0), len(y0))
        adj_all = adj_all + adj0
        degree_all = degree_all + degree0
        y_all = y_all + y0
        print(len(adj_all), len(degree_all), len(y_all))
    print('process data success')

    city_data = Data(x=degree_all, y=y_all, edge_index=adj_all)
    print(f'city_data: {city_data}')
    # print(f'x:  {city_data.x}')
    # print(f'Edge:  {city_data.edge_index.t()}')
    city_dataset = MyOwnDataset(save_root="city")  # 100个样本（图）#
    data = Batch.from_data_list(city_dataset)

    wl = WL(num_layers=5)
    import pdb;pdb.set_trace()
    hists = wl(data.x, data.edge_index, data.batch)

    test_accs = torch.empty(args.runs, dtype=torch.float)

    for run in range(1, args.runs + 1):
        perm = torch.randperm(data.num_graphs)
        val_index = perm[:data.num_graphs // 10]
        test_index = perm[data.num_graphs // 10:data.num_graphs // 5]
        train_index = perm[data.num_graphs // 5:]

        best_val_acc = 0

        for hist in hists:
            train_hist, train_y = hist[train_index], data.y[train_index]
            val_hist, val_y = hist[val_index], data.y[val_index]
            test_hist, test_y = hist[test_index], data.y[test_index]

            for C in [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2, 10**-3]:
                model = LinearSVC(C=C, tol=0.01, dual=True)
                model.fit(train_hist, train_y)
                val_acc = accuracy_score(val_y, model.predict(val_hist))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = accuracy_score(test_y, model.predict(test_hist))
                    test_accs[run - 1] = test_acc

        print(f'Run: {run:02d}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

    print(f'Final Test Performance: {test_accs.mean():.4f}±{test_accs.std():.4f}')
