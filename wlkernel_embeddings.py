import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
import os
from sklearn.manifold import TSNE
from torch_geometric.nn import WLConv
import torch
import pandas as pd
from torch_geometric.data import Data
import numpy as np
from torch_geometric.data import Batch
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore', category=ConvergenceWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
torch.manual_seed(42)

CLASS = 3


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
        if edge_index2.shape[1] < 20:
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
        city_dataset.extend(get_city_dataset(i, cities[i][1]))
        # print(city_dataset)

    data = Batch.from_data_list(city_dataset)
    print('process data success')
    # ================================================================================
    # print(f'city_dataset[0]: {city_dataset[0]}')
    wl = WL(num_layers=5)
    hists = wl(data.x, data.edge_index, data.batch)
    tsne = TSNE(n_components=2, random_state=42)
    colormap = plt.get_cmap('coolwarm')
    colors = colormap(np.linspace(0, 1, CLASS))
    for _index, hist in enumerate(hists):
        embedding = tsne.fit_transform(hist)
        for i in range(CLASS):
            plt.scatter(embedding[np.where(data.y == i), 0], embedding[np.where(data.y == i), 1],
                        color=colors[i], s=10, linewidths=0., label=cities[i][1])
        # plt.legend()  # 图例的位置 loc='upper right'
        plt.savefig(pwd + f"{_index} embedding.png")
        plt.show()
        plt.close()
