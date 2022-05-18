import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from GCNmodel import GCNModel, GATModel, EGATModel
from dataset import YeastDataset
from path_utils import PATH


def evaluate(model, graph_list, dataset, yeast_data):
    model.eval()
    with torch.no_grad():
        edge_features = yeast_data.edge_features2[100]
        graph = graph_list[100]
        node_in_degrees = dataset.node_in_degrees[100]
        node_out_degrees = dataset.node_out_degrees[100]
        node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
        edge_labels = dataset.edge_labels[100]
        logits = model(graph, node_features, edge_features)
        pred = logits.max(1).indices
        loss = F.cross_entropy(logits, edge_labels)
        # print("Eval acc: " + str(torch.sum(pred == edge_labels) / len(edge_labels)))
        # print("Eval F1: " + str(f1_score(edge_labels, pred, average="macro")))
        return loss, torch.sum(pred == edge_labels), f1_score(edge_labels, pred, average="macro")


if __name__ == "__main__":

    # torch.set_num_threads(32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter()

    yeast_dataset = YeastDataset(PATH, PATH)

    if yeast_dataset.has_cache():
        yeast_dataset.load()
    else:
        yeast_dataset.process()
        yeast_dataset.save()

    graph_list = yeast_dataset.graph_list

    # model = GCNModel(2, 1, 128, 64, 64, 2)
    # model = GATModel(2, 1, 128, 512, 256, 2, 3)
    model = EGATModel(2, 1, 128, 512, 256, 2, 3)
    # if device == 'cuda':
    #     model = model.to(device)
    opt = torch.optim.Adam(model.parameters())

    max_f1 = 0

    for epoch in range(500):
        print("Epoch: " + str(epoch))
        for i, graph in enumerate(graph_list[:100]):
            print(i)
            node_in_degrees = yeast_dataset.node_in_degrees[i]
            node_out_degrees = yeast_dataset.node_out_degrees[i]
            node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
            # node_features = node_features.to(device)
            edge_features = yeast_dataset.edge_features2[i]
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
        loss, acc, f1 = evaluate(model, graph_list, yeast_dataset, yeast_dataset)

        if f1 > max_f1:
            print("Eval acc: " + str(acc / len(edge_labels)))
            print("Eval F1: " + str(f1))
            print("Eval loss: " + str(loss))

        max_f1 = max(f1, max_f1)

        writer.add_scalar('Loss/val', loss, epoch)
        writer.add_scalar('Accuracy/val', acc / len(edge_labels), epoch)
        writer.add_scalar('F1/val', f1, epoch)
