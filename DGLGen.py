import pandas as pd
import dgl
import torch
import torch.nn.functional as F


nodes_data = pd.read_csv('/home/filip/Desktop/yeast_data/graphs/graph_nodes.csv')
print(nodes_data)

edges_data = pd.read_csv('/home/filip/Desktop/yeast_data/graphs/graph_edges.csv')
print(edges_data)

src = edges_data['node1'].to_numpy()
dst = edges_data['node2'].to_numpy()

g = dgl.graph((src, dst))
bg = dgl.to_simple(g)
bg = dgl.to_bidirected(bg)

print('#Nodes', bg.number_of_nodes())
print('#Edges', bg.number_of_edges())

mut = nodes_data['node_class'].to_numpy()
mut_tensor = torch.tensor(mut)

mut_onehot = F.one_hot(mut_tensor)

bg.ndata.update({'mut': mut_tensor, 'mut_onehot': mut_onehot})

print(bg)
