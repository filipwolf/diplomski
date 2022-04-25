import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from GCNmodel import evaluate, GCNModel, GATModel
from dataset import YeastDataset

if __name__ == "__main__":

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yeast_dataset = YeastDataset('/media/filip/DA2A5AE02A5AB8E92/diplomski/yeast_data/graphs/',
                                 '/media/filip/DA2A5AE02A5AB8E92/diplomski/yeast_data/graphs/')

    if yeast_dataset.has_cache():
        yeast_dataset.load()
    else:
        yeast_dataset.process()
        yeast_dataset.save()

    graph_list = yeast_dataset.graph_list

    model = GCNModel(2, 1, 128, 64, 64, 2)
    # model = GATModel(2, 1, 128, 512, 256, 2, 3)
    # model = model.to(device)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(50):
        print('Epoch: ' + str(epoch))
        for i, graph in enumerate(graph_list[:100]):
            node_in_degrees = yeast_dataset.node_in_degrees[i]
            node_out_degrees = yeast_dataset.node_out_degrees[i]
            node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
            # node_features = node_features.to(device)
            edge_features = yeast_dataset.edge_features[i]
            # edge_features = edge_features.resize(len(edge_features), 1)
            # edge_features = edge_features.resize(19841, 1)
            edge_labels = yeast_dataset.edge_labels[i]
            logits = model(graph, node_features, edge_features)

            pred = torch.softmax(logits, dim=1).max(1).indices
            loss = F.cross_entropy(logits, edge_labels)
            # compute validation accuracy
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print('Train loss: ' + str(loss.item()))
            # print('Train acc: ' + str(torch.sum(pred == edge_labels)/len(edge_labels)))
            # print('Train F1: ' + str(f1_score(edge_labels, pred)))
        acc = evaluate(model, graph_list, yeast_dataset, yeast_dataset.edge_features[100])
        print('Eval loss: ' + str(acc))
