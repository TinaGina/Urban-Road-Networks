import os
import timeit
import networkx as nx
import osmnx as ox
import pandas as pd
import torch

ox.config(use_cache=True, log_console=True)  # 多个城市的时候需要缓存


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('A new folder created.')
    else:
        print('Has already been created.')


def adjacency_matrix(city):
    pwd = './Mydata0120/'
    mkdir(pwd)
    lat_center = float(city[3])
    lon_center = float(city[4])
    lat_orig = lat_center - 0.1  # 找到初始点
    lon_orig = lon_center - 0.1  # 找到初始点
    # custom_filter = '["railway"]'
    side = 0.01  # 每个区间都是经纬度为0.01的方形
    n = 20  # 区间横20，纵20，应该设为20
    lat = lat_orig
    lon = lon_orig
    for i in range(n):
        for j in range(n):
            try:
                G1 = ox.graph_from_bbox(lat+side, lat, lon+side, lon, network_type='drive', simplify=True)
                G1 = ox.project_graph(G1)
                # fig, ax = ox.plot_graph(G_projected)
                adj1 = nx.to_numpy_array(G1)
                adj1 = torch.from_numpy(adj1)
                edge_index1 = torch.nonzero(adj1).transpose(0, 1)
                edge_index_subfile = pwd + 'edge_index' + city[1] + 'lat' + str(i) + 'lon' + str(j) + '.csv'
                edge_index_df = pd.DataFrame(edge_index1)
                edge_index_df.to_csv(edge_index_subfile, index=False)

                G1nodes = list(G1.nodes(data=True))
                x1 = torch.tensor([[G1nodes[i1][1]['lon'], G1nodes[i1][1]['lat']] for i1 in range(len(G1nodes))])
                x_subfile = pwd + 'x' + city[1] + 'lat' + str(i) + 'lon' + str(j) + '.csv'
                x_df = pd.DataFrame(x1)
                x_df.to_csv(x_subfile, index=False)
            except Exception as e:
                print(e)
            print('lat, lon', lat, lon)
            lon = lon + side
        lat = lat + side
        lon = lon_orig


if __name__ == '__main__':
    cities = []
    f = open("./world_city_20240217.txt", "r")
    for line in f.readlines():
        linestr = line.strip()
        linestrlist = linestr.split("\t")
        cities.append(linestrlist)

    # start0 = timeit.default_timer()
    # countReal = 0
    # count = 0
    # for i in range(len(cities)):
    #     print('===========================================')
    #     print(cities[i][0], cities[i][1])
    #     adjacency_matrix(cities[i])

    # 单独更新了Toronto和Chicago
    print(cities[11][0], cities[11][1])
    adjacency_matrix(cities[11])

    print(cities[18][0], cities[18][1])
    adjacency_matrix(cities[18])



