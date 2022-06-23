import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

from GCNmodel import GCNModel, GATModel, EGATModel
from dataset import YeastDataset
from path_utils import PATH
from scripts.graphiaGen2 import gen


def evaluate(model, graph_list, yeast_data, index):
    """Function used for evaluating model performance.

    Parameters
    ----------
    model : nn.Module
        The model used for training.
    graph_list : list
       List of all graphs used for training.
    yeast_data : YeastDataset
        The dataset we defined for training, used for fetching various graph features.
    """
    model.eval()
    with torch.no_grad():
        # load edge features
        edge_features = yeast_data.edge_features2[index]
        # select graph for validation
        graph = graph_list[index]
        # load node features
        node_in_degrees = yeast_data.node_in_degrees[index]
        node_out_degrees = yeast_data.node_out_degrees[index]
        node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
        # load edge labels
        edge_labels = yeast_data.edge_labels[index]
        # calculate model outputs
        logits = model(graph, node_features, edge_features)
        pred = logits.max(1).indices
        # calculate loss
        loss = F.cross_entropy(logits, edge_labels)
        # print("Eval acc: " + str(torch.sum(pred == edge_labels) / len(edge_labels)))
        # print("Eval F1: " + str(f1_score(edge_labels, pred, average="macro")))
        return (
            loss,
            torch.sum(pred == edge_labels),
            f1_score(edge_labels, pred, average="macro"),
            precision_score(edge_labels, pred),
            recall_score(edge_labels, pred),
            pred,
        )


if __name__ == "__main__":

    # select number of threads to be used
    # torch.set_num_threads(32)

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init writer for storing paramters to TensorBoard
    writer = SummaryWriter()

    # initiate dataset
    yeast_dataset = YeastDataset(PATH, PATH)

    # load dataset or generate if it doesn't exist yet
    if yeast_dataset.has_cache():
        yeast_dataset.load()
    else:
        yeast_dataset.process()
        yeast_dataset.save()

    # load list of graphs
    graph_list = yeast_dataset.graph_list

    # initiate model
    model = GCNModel(2, 1, 128, 64, 64, 2)
    # model = GATModel(2, 1, 128, 512, 256, 2, 3)
    # model = EGATModel(2, 1, 64, 256, 128, 2, 3)

    # move mode to GPU if it's available
    # if device == 'cuda':
    #     model = model.to(device)

    # initiate optimizer
    opt = torch.optim.Adam(model.parameters())

    max_f1 = 0
    best_predictions = 0

    # training loop
    for epoch in range(1):
        print("Epoch: " + str(epoch))
        for i, graph in enumerate(graph_list[:100]):
            # load node features
            node_in_degrees = yeast_dataset.node_in_degrees[i]
            node_out_degrees = yeast_dataset.node_out_degrees[i]
            node_features = torch.transpose(torch.stack((node_in_degrees, node_out_degrees)), 0, 1)
            # node_features = node_features.to(device)
            # load edge features
            edge_features = yeast_dataset.edge_features2[i]
            # edge_features = edge_features.resize(19841, 1)
            # load edge labels
            edge_labels = yeast_dataset.edge_labels[i]
            # calculate model outputs
            logits = model(graph, node_features, edge_features)

            # pred = torch.softmax(logits, dim=1).max(1).indices
            # calculate loss
            loss = F.cross_entropy(logits, edge_labels)
            # calculate gradients
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print('Train loss: ' + str(loss.item()))
            # print('Train acc: ' + str(torch.sum(pred == edge_labels)/len(edge_labels)))
            # print('Train F1: ' + str(f1_score(edge_labels, pred)))
        # validate model
        loss, acc, f1, precision, recall, predictions = evaluate(model, graph_list, yeast_dataset, 101)

        # save best model so far
        if f1 > max_f1:
            print("Eval acc: " + str(acc / len(yeast_dataset.edge_labels[101])))
            print("Eval F1: " + str(f1))
            print("Eval loss: " + str(loss))
            print("Eval precision: " + str(precision))
            print("Eval recall: " + str(recall))
            gen(predictions)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "f1": f1,
                },
                "outputs/best_model.pth",
            )

        max_f1 = max(f1, max_f1)

        # add data to TensorBoard
        writer.add_scalar("Loss/val", loss, epoch)
        writer.add_scalar("Accuracy/val", acc / len(yeast_dataset.edge_labels[101]), epoch)
        writer.add_scalar("F1/val", f1, epoch)
        writer.add_scalar("precision/val", precision, epoch)
        writer.add_scalar("recall/val", recall, epoch)
