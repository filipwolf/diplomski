import os

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl import save_graphs, load_graphs
from dgl.data import CoraGraphDataset
from dgl.data import DGLDataset
from dgl.data.utils import save_info, load_info


class YeastDataset(DGLDataset):
    """Template for customizing graph datasets in DGL.

    Parameters
    ----------
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self, raw_dir=None, save_dir=None, force_reload=False, verbose=False):
        super(YeastDataset, self).__init__(
            name="dataset_name", raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload, verbose=verbose
        )
        self.edge_features2 = None
        self.edge_labels = None
        self.node_labels = None
        self.node_features = None
        self.node_in_degrees = None
        self.node_out_degrees = None
        self.graph_list = None
        self.edge_features = None
        self.num_edges = None
        self.num_nodes = None

    def process(self):

        self.graph_list = []
        self.num_nodes = []
        self.num_edges = []
        self.node_out_degrees = []
        self.node_in_degrees = []
        self.node_labels = []
        self.edge_features = []
        self.edge_features2 = []
        self.edge_labels = []

        for i in range(0, 101):
            print(i)
            nodes_data = pd.read_csv(self.raw_dir + "node_features/node_features" + str(i) + ".csv")
            edges_data = pd.read_csv(self.raw_dir + "edge_features/edge_features" + str(i) + ".csv")

            src = edges_data["node1"].to_numpy()
            dst = edges_data["node2"].to_numpy()
            lengths = edges_data["edge_length"].to_numpy()
            overlap = edges_data["edge_overlap"]
            lengths_tensor = torch.FloatTensor(lengths)
            overlaps_tensor = torch.FloatTensor(overlap)

            g = dgl.graph((src, dst))

            g.edata["edge_lengths"] = lengths_tensor
            g.edata["edge_overlaps"] = overlaps_tensor

            mut = nodes_data["node_class"].to_numpy()
            mut_tensor = torch.tensor(mut)

            mut_onehot = F.one_hot(mut_tensor)

            g.ndata.update({"mut_tensor": mut_tensor, "mut_onehot": mut_onehot})

            edge_labels = edges_data["edge_class"].to_numpy()
            edge_tensor = torch.tensor(edge_labels)
            edge_tensor_onehot = F.one_hot(edge_tensor)
            g.edata.update({"mut_tensor": edge_tensor, "mut_onehot": edge_tensor_onehot})
            g = dgl.add_self_loop(g)
            g.edata["edge_lengths"][g.edata["edge_lengths"] == 0] = 1

            self.num_nodes.append(g.number_of_nodes())
            self.num_edges.append(g.number_of_edges())

            node_out_degree_list = []
            node_in_degree_list = []

            for j in range(self.num_nodes[i]):
                node_out_degree_list.append(g.out_degrees(j))
                node_in_degree_list.append(g.in_degrees(j))

            node_out_degree_list = np.array(node_out_degree_list)
            node_in_degree_list = np.array(node_in_degree_list)
            node_out_degree_list_tensor = torch.FloatTensor(node_out_degree_list)
            node_in_degree_list_tensor = torch.FloatTensor(node_in_degree_list)

            self.node_out_degrees.append(node_out_degree_list_tensor)
            self.node_in_degrees.append(node_in_degree_list_tensor)
            self.node_labels.append(mut_tensor)
            self.edge_labels.append(g.edata["mut_tensor"])
            self.edge_features.append(g.edata["edge_lengths"])
            self.edge_features2.append(g.edata["edge_overlaps"])

            g.ndata.update(
                {"node_out_degrees": node_out_degree_list_tensor, "node_in_degrees": node_in_degree_list_tensor}
            )

            self.graph_list.append(g)

    def __getitem__(self, idx):
        return self.graph_list[idx]

    def __len__(self):
        return len(self.graph_list)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_dir + "dgl_graph.bin")
        save_graphs(graph_path, self.graph_list)
        # save other information in python dict
        info_path = os.path.join(self.save_dir, "info.pkl")
        save_info(
            info_path,
            {
                "node_in_degrees": self.node_in_degrees,
                "node_out_degrees": self.node_out_degrees,
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
                "edge_features": self.edge_features,
                "edge_overlaps": self.edge_features2,
                "edge_labels": self.edge_labels,
            },
        )

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_dir + "dgl_graph.bin")
        self.graph_list, _ = load_graphs(graph_path)
        info_path = os.path.join(self.save_dir, "info.pkl")
        self.node_in_degrees = load_info(info_path)["node_in_degrees"]
        self.node_out_degrees = load_info(info_path)["node_out_degrees"]
        self.num_nodes = load_info(info_path)["num_nodes"]
        self.num_edges = load_info(info_path)["num_edges"]
        self.edge_features = load_info(info_path)["edge_features"]
        self.edge_features2 = load_info(info_path)["edge_overlaps"]
        self.edge_labels = load_info(info_path)["edge_labels"]

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_dir, "dgl_graph.bin")
        info_path = os.path.join(self.save_dir, "info.pkl")
        return os.path.exists(graph_path) and os.path.exists(info_path)


def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    return g, features, labels, train_mask, test_mask
