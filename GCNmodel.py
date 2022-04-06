import dgl
import dgl.function as fn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class Model(nn.Module):
    def __init__(self, node_features, edge_features, lin_dim, hidden_dim, out_dim, n_classes):
        super(Model, self).__init__()
        self.lin_n = nn.Linear(node_features, lin_dim)
        #self.lin_e = nn.Linear(edge_features, lin_dim)
        self.conv1 = GraphConv(lin_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)
        self.classify = DotProductPredictor()

    def forward(self, graph, node_features, edge_features):
        node_f = self.lin_n(node_features)
        #edge_f = self.lin_e(edge_features)
       # cat_features = torch.stack((node_f, edge_f))
        h = F.relu(self.conv1(graph, node_f))
        h = F.relu(self.conv2(graph, h))
        h = self.classify(graph, h)
        return h


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        graph.ndata['h'] = h
        graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        return graph.edata['score']


def evaluate(model, graph_list, dataset):
    model.eval()
    with torch.no_grad():
        graph = dgl.add_self_loop(graph_list[11])
        node_in_degrees = dataset.node_in_degrees[11]
        node_out_degrees = dataset.node_out_degrees[11]
        node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
        edge_labels = dataset.edge_labels[11]
        logits = model(graph, node_features, 0)
        loss = ((logits - edge_labels) ** 2).mean()
        return loss


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()
