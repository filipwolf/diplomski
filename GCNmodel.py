import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import EdgeWeightNorm, GATConv
from dgl.nn.pytorch.conv import GraphConv
from sklearn.metrics import f1_score


class GCNModel(nn.Module):
    def __init__(self, node_features, edge_features, lin_dim, hidden_dim, out_dim, n_classes):
        super(GCNModel, self).__init__()
        self.lin_n = nn.Linear(node_features, lin_dim)
        self.lin_e = nn.Linear(edge_features, lin_dim)
        self.conv1 = GraphConv(lin_dim, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, out_dim)
        self.classify = MLPPredictor(out_dim, n_classes)
        self.dp = nn.Dropout(p=0.3)
        self.norm = EdgeWeightNorm(norm='right')

    def forward(self, graph, node_features, edge_features):
        norm_edge_weight = self.norm(graph, edge_features)
        node_f = self.lin_n(node_features)
        # edge_f = self.lin_e(edge_features)
        # cat_features = torch.stack((node_f, edge_f))
        h = self.dp(F.relu(self.conv1(graph, node_f)))
        h = self.dp(F.relu(self.conv2(graph, h)))
        h = self.dp(F.relu(self.conv3(graph, h)))
        h = self.dp(F.relu(self.conv4(graph, h)))
        h = self.classify(graph, h)
        return h


class GATModel(nn.Module):
    def __init__(self, node_features, edge_features, lin_dim, hidden_dim, out_dim, n_classes, num_heads):
        super(GATModel, self).__init__()
        self.lin_n = nn.Linear(node_features, lin_dim)
        self.lin_e = nn.Linear(edge_features, lin_dim)
        self.gat1 = GATConv(lin_dim, hidden_dim, num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, out_dim, 1)
        self.conv1 = GraphConv(out_dim, int(out_dim / 2))
        self.conv2 = GraphConv(int(out_dim / 2), int(out_dim / 4))
        self.classify = MLPPredictor(int(out_dim / 4), n_classes)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, graph, h):
        node_f = self.lin_n(h)
        h = torch.flatten(self.dp(F.elu(self.gat1(graph, node_f))), start_dim=1)
        h = torch.flatten(self.dp(F.elu(self.gat2(graph, h))), start_dim=1)
        h = self.dp(F.elu(self.conv1(graph, h)))
        h = self.dp(F.elu(self.conv2(graph, h)))
        h = self.classify(graph, h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        graph.ndata['h'] = h
        graph.apply_edges(self.apply_edges)
        return graph.edata['score']


def evaluate(model, graph_list, dataset, edge_features):
    model.eval()
    with torch.no_grad():
        graph = graph_list[100]
        node_in_degrees = dataset.node_in_degrees[100]
        node_out_degrees = dataset.node_out_degrees[100]
        node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
        edge_labels = dataset.edge_labels[100]
        logits = model(graph, node_features)
        pred = logits.max(1).indices
        loss = F.cross_entropy(logits, edge_labels)
        print('Eval acc: ' + str(torch.sum(pred == edge_labels) / len(edge_labels)))
        print('Eval F1: ' + str(f1_score(edge_labels, pred)))
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
