import os

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
import torch.nn.functional as F
from dgl.data.citation_graph import _sample_mask
from dgl.data.knowledge_graph import _read_triplets_as_list, build_knowledge_graph
from dgl.data.utils import generate_mask_tensor
from dgl.data import CoraGraphDataset


class YeastDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

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

    def __init__(self,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(YeastDataset, self).__init__(name='dataset_name',
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)
        self.test_mask = None
        self.train_mask = None
        self.labels = None
        self.node_degrees = None
        self.num_edges = None
        self.num_nodes = None
        self.g = None

    def process(self):
        # """
        # The original knowledge base is stored in triplets.
        # This function will parse these triplets and build the DGLGraph.
        # """
        # root_path = self.raw_path
        # entity_path = os.path.join(root_path, 'graph_nodes.csv')
        # relation_path = os.path.join(root_path, 'graph_edges.csv')
        # train_path = os.path.join(root_path, 'train.txt')
        # valid_path = os.path.join(root_path, 'valid.txt')
        # test_path = os.path.join(root_path, 'test.txt')
        # nodes_data = pd.read_csv(entity_path)
        # edges_data = pd.read_csv(relation_path)
        # train = np.asarray(_read_triplets_as_list(train_path, nodes_data, edges_data))
        # valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        # test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        # num_nodes = len(entity_dict)
        # num_rels = len(relation_dict)
        # if self.verbose:
        #     print("# entities: {}".format(num_nodes))
        #     print("# relations: {}".format(num_rels))
        #     print("# training edges: {}".format(train.shape[0]))
        #     print("# validation edges: {}".format(valid.shape[0]))
        #     print("# testing edges: {}".format(test.shape[0]))
        #
        # # for compatability
        # self._train = train
        # self._valid = valid
        # self._test = test
        #
        # self._num_nodes = num_nodes
        # self._num_rels = num_rels
        # # build graph
        # g, data = build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=self.reverse)
        # etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, val_mask, test_mask = data
        # g.edata['train_edge_mask'] = train_edge_mask
        # g.edata['valid_edge_mask'] = valid_edge_mask
        # g.edata['test_edge_mask'] = test_edge_mask
        # g.edata['train_mask'] = train_mask
        # g.edata['val_mask'] = val_mask
        # g.edata['test_mask'] = test_mask
        # g.edata['etype'] = etype
        # g.ndata['ntype'] = ntype
        # self._g = g

        nodes_data = pd.read_csv(self.raw_dir + 'graph_nodes.csv')

        edges_data = pd.read_csv(self.raw_dir + 'graph_edges.csv')

        src = edges_data['node1'].to_numpy()
        dst = edges_data['node2'].to_numpy()

        g = dgl.graph((src, dst))
        sg = dgl.to_simple(g)
        g = dgl.to_bidirected(sg)

        mut = nodes_data['node_class'].to_numpy()
        mut_tensor = torch.tensor(mut)

        mut_onehot = F.one_hot(mut_tensor)

        g.ndata.update({'mut_tensor': mut_tensor, 'mut_onehot': mut_onehot})

        self.num_nodes = g.number_of_nodes()
        self.num_edges = g.number_of_edges()

        node_degree_list = []

        for i in range(self.num_nodes):
            node_degree_list.append(g.out_degrees(i))

        node_degree_list = np.array(node_degree_list)
        node_degree_list_tensor = torch.tensor(node_degree_list)

        self.node_degrees = node_degree_list_tensor
        self.labels = mut_tensor

        g.ndata.update({'node_degrees': node_degree_list_tensor})

        idx_train = range(len(mut_onehot) - 500)
        idx_test = range(len(mut_onehot) - 500, len(mut_onehot))

        train_mask = generate_mask_tensor(_sample_mask(idx_train, len(mut_onehot)))
        test_mask = generate_mask_tensor(_sample_mask(idx_test, len(mut_onehot)))
        self.train_mask = train_mask
        self.test_mask = test_mask

        # # splitting mask
        # g.edata['train_mask'] = train_mask
        # g.edata['val_mask'] = val_mask
        # g.edata['test_mask'] = test_mask
        # # edge type
        # g.edata['etype'] = etype
        # # node type
        # g.ndata['ntype'] = ntype
        self.g = g

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.g

    def __len__(self):
        return 1

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path + '_dgl_graph.bin')
        save_graphs(graph_path, self.g)

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path + '_dgl_graph.bin')
        self.g = load_graphs(graph_path)

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path + '_dgl_graph.bin')
        return os.path.exists(graph_path)


def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    return g, features, labels, train_mask, test_mask
