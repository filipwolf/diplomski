import time
import numpy as np
import torch

from GCNmodel import evaluate, construct_negative_graph, compute_loss, Model
from dataset import YeastDataset, load_cora_data
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader

if __name__ == "__main__":
    yeast_dataset = YeastDataset('/media/filip/DA2A5AE02A5AB8E9/diplomski/yeast_data/graphs/',
                                 '/media/filip/DA2A5AE02A5AB8E9/diplomski/yeast_data/graphs/')

    yeast_dataset.process()
    graph_list = yeast_dataset.graph_list
    node_features = yeast_dataset.node_out_degrees[0]
    edge_features = yeast_dataset.edge_features[0]
    n_labels = 2

    model = Model(2, 1, 32, 64, 32, 2)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        for i, graph in enumerate(graph_list[:10]):
            graph = dgl.add_self_loop(graph)
            node_in_degrees = yeast_dataset.node_in_degrees[i]
            node_out_degrees = yeast_dataset.node_out_degrees[i]
            node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
            #edge_features = yeast_dataset.edge_features[i]
            #edge_features = edge_features.resize(19841, 1)
            edge_labels = yeast_dataset.edge_labels[i]
            model.train()
            logits = model(graph, node_features, edge_features)
            loss = ((logits - edge_labels)**2).mean()
            # compute validation accuracy
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())
        acc = evaluate(model, graph_list, yeast_dataset)
        print('Eval acc: ' + acc)
