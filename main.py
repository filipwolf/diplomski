import time
import numpy as np
import torch

from GCNmodel import Net, evaluate, construct_negative_graph, compute_loss
from dataset import YeastDataset, load_cora_data
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

if __name__ == "__main__":

    # yeast_dataset = YeastDataset('/home/filip/Desktop/yeast_data/graphs/', '/home/filip/Desktop/yeast_data/graphs/')
    # yeast_dataset.process()
    # print(yeast_dataset.num_nodes)
    # print(yeast_dataset.num_edges)
    # g = yeast_dataset.g
    # features = yeast_dataset.node_degrees
    # labels = yeast_dataset.labels
    # train_mask = yeast_dataset.train_mask
    # test_mask = yeast_dataset.test_mask

    dataset = dgl.data.CiteseerGraphDataset()
    graph = dataset[0]
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)

    k = 5
    model = Net(n_features, 100)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        negative_graph = construct_negative_graph(graph, k)
        pos_score, neg_score = model(graph, negative_graph, node_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
