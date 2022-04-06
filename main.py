import time
import numpy as np
import torch

from GCNmodel import Classifier, evaluate, construct_negative_graph, compute_loss
from dataset import YeastDataset, load_cora_data
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader


if __name__ == "__main__":


    yeast_dataset = YeastDataset('/home/filip/Desktop/yeast_data/graphs/', '/home/filip/Desktop/yeast_data/graphs/')

    graph_list = []

    yeast_dataset.process()
    graph = yeast_dataset.g
    node_features = yeast_dataset.node_degrees
    node_labels = yeast_dataset.labels
    n_features = len(node_features)
    n_labels = 2

    model = Classifier(in_dim=n_features, hidden_dim=100, n_classes=n_labels)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        for graph in graph_list:
            model.train()
            # forward propagation by using all nodes
            logits = model(graph, node_features)
            # compute loss
            loss = F.cross_entropy(logits, node_labels)
            # compute validation accuracy
            acc = evaluate(model, graph, node_features, node_labels)
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())
