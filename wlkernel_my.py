import argparse
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
import os

from sklearn.model_selection import train_test_split
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
import networkx as nx
from grakel.utils import graph_from_networkx
import grakel

warnings.filterwarnings('ignore', category=ConvergenceWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
torch.manual_seed(42)

CLASS = 2


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


class RW(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.output = None
        self.hidden_activation = None
        self.rw = grakel.RandomWalk(normalize=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.biases_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.biases_output = np.zeros((1, self.output_size))

    def forward(self, x):
        output = self.rw.fit_transform(x)
        return output

    def transform(self, x):
        output = self.rw.transform(x)
        return output


class SP(torch.nn.Module):
    def __init__(self):
        self.output = None
        self.hidden_activation = None
        self.rw = grakel.GraphKernel(kernel='SP', normalize=False, verbose=False, n_jobs=None, random_state=None,
                                     Nystroem=False)

    def forward(self, x):
        output = self.rw.fit_transform(x)
        return output

    def transform(self, x):
        output = self.rw.transform(x)
        return output


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def get_city_dataset(num, cityname):
    filenames = os.listdir('./Mydata')
    filenames = [x1 for x1 in filenames if x1[:len(cityname)] == cityname]
    city_dataset1 = []
    label1 = []
    for file in filenames:
        adj1 = pd.read_csv(os.path.join('./Mydata', file))
        print("正在处理", file)
        G = nx.Graph()
        G.add_nodes_from(list(range(adj1.shape[0])))
        for i in range(adj1.shape[0] - 1):
            for j in range(i + 1, adj1.shape[0]):
                if adj1.iloc[i, j] == 1:
                    G.add_edge(i, j)
        city_dataset1.append(G)
        label1.append(num)
    return city_dataset1, label1


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
    label = []
    for i in range(0, CLASS):  # 修改为 len(cities)
        print('===================')
        city_dataset1, label1 = get_city_dataset(cities[i][0], cities[i][1])
        city_dataset.extend(city_dataset1)
        label.extend(label1)
        print(city_dataset)

    print('process data success')
    # ================================================================================
    print(f'city_dataset[0]: {city_dataset[0]}')
    G = graph_from_networkx(city_dataset)
    # G_train, G_test, y_train, y_test = train_test_split(G, label, test_size=0.2, random_state=42)

    rw = RW(input_size=3, hidden_size=5, output_size=5)
    _start = time.time()
    hists = rw.forward(G)
    print(f"用时{(time.time() - _start) / 1000}s")

    length = len(city_dataset)
    perm = torch.randperm(length)
    test_index = perm[:length // 5]
    train_index = perm[length // 5:]

    train_hist, train_y = hists[train_index], label[train_index]
    test_hist, test_y = hists[test_index], label[test_index]
    for C in [10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]:
        model = LinearSVC(C=C, tol=0.01, dual=True)
        model.fit(train_hist, train_y)
        K_test = rw.transform(test_hist)
        val_acc = accuracy_score(test_y, model.predict(K_test))
        print(val_acc)
