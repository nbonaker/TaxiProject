import dgl
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
import numpy as np
import pandas as pd
import datetime
import os
import time
from matplotlib import pyplot as plt


def create_dgl_graphs(dir):
    edge_data_path = dir + "scipy_graphs/"

    node_data_by_month = [pd.read_csv(dir+"surge_2019-0"+str(i)+".csv") for i in range(1, 7)]
    for df in node_data_by_month:
        df["interval_datetime"] = pd.to_datetime(df["interval_datetime"], format='%Y-%m-%d %H:%M:%S',
                                                          errors='ignore')

    dgl_graphs = []
    i = 0
    for graph_file in os.listdir(edge_data_path):
        sparse_adj = sparse.load_npz(edge_data_path + graph_file)
        weights = th.tensor(list(sparse_adj.data), dtype=th.int32)

        month_num, interval_num = graph_file.split("-")
        month_num = int(month_num.split("_")[-1])
        interval_num = int(interval_num.split(".")[0])-1

        cur_interval_start = datetime.datetime(2019, month_num, 1, 0, 00, 0) + datetime.timedelta(0, 10*60*interval_num)
        node_DF = node_data_by_month[month_num-1][node_data_by_month[month_num-1]["interval_datetime"] == cur_interval_start]

        label_DF = node_data_by_month[month_num - 1][
            node_data_by_month[month_num - 1]["interval_datetime"] == cur_interval_start + datetime.timedelta(0, 10*60)]
        node_labels = th.from_numpy(label_DF[label_DF.columns[9:]].values.astype(int).T)

        node_base_features = node_DF[['is_holiday', "PU_time_2AM", "PU_time_6AM", "PU_time_10AM",
                                             "PU_time_2PM", "PU_time_6PM", "PU_time_10PM"]].values.astype(int)[0]
        node_surge_features = node_DF[node_DF.columns[9:]].values.astype(int)

        node_base_features = np.array([node_base_features for i in range(node_surge_features.size)])
        node_features = th.from_numpy(np.vstack([node_surge_features, node_base_features.T]).T)

        g = dgl.from_scipy(sparse_adj)
        g.edata['feature'] = weights
        g.ndata['feature'] = node_features
        g.ndata['label'] = node_labels

        train_mask = np.random.randint(0, 10, size=len(node_labels))
        test_mask = np.where(train_mask == -1, 1, 0).astype(bool)
        val_mask = np.where(train_mask == -1, 1, 0).astype(bool)
        train_mask = np.where(train_mask > 0, 1, 0).astype(bool)

        g.ndata['train_mask'] = th.from_numpy(train_mask)
        g.ndata['test_mask'] = th.from_numpy(test_mask)
        g.ndata['val_mask'] = th.from_numpy(val_mask)
        g.add_edges(g.nodes(), g.nodes())

        dgl_graphs.append(g)

        i += 1
        if i > 100:
            break

    np.random.shuffle(dgl_graphs)
    return dgl_graphs


def main():

    dataset = create_dgl_graphs("surge_prediction_data/")
    print(dataset[0])

    class SAGE(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats):
            super().__init__()
            self.conv1 = dglnn.SAGEConv(
                in_feats=in_feats, out_feats=hid_feats, aggregator_type='pool')
            self.conv2 = dglnn.SAGEConv(
                in_feats=hid_feats, out_feats=1000, aggregator_type='pool')
            self.conv3 = dglnn.SAGEConv(
                in_feats=1000, out_feats=out_feats, aggregator_type='pool')

        def forward(self, graph, inputs):
            inputs = inputs.float()
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = F.relu(h)
            h = self.conv2(graph, h)
            h = F.relu(h)
            h = self.conv3(graph, h)
            return h

    def evaluate(model, graph, features, labels, mask):
        model.eval()
        with th.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = th.max(logits, dim=1)
            correct = th.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    graph = dataset[0]

    node_features = graph.ndata['feature'].float()
    node_labels = graph.ndata['label'].float()
    # train_mask = graph.ndata['train_mask']
    # valid_mask = graph.ndata['val_mask']
    # test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = 1

    model = SAGE(in_feats=n_features, hid_feats=500, out_feats=n_labels)
    opt = th.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1):
        for graph_num, graph in enumerate(dataset):
            node_features = graph.ndata['feature'].float()
            node_labels = graph.ndata['label'].float()

            model.train()
            # forward propagation by using all nodes
            logits = model(graph, node_features)
            # compute loss
            loss = F.mse_loss(logits, node_labels)
            # compute validation accuracy
            # acc = evaluate(model, graph, node_features, node_labels, valid_mask)
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            # if graph_num % 10 == 0:
            print(loss.item())

    test_graph = dataset[-1]
    node_features = graph.ndata['feature'].float()
    node_labels = graph.ndata['label'].float()

    logits = model(graph, node_features)
    preds = list(logits.detach().numpy().T[0])
    targets = list(node_labels.detach().numpy().T[0])

    combined = list(zip(preds, targets))
    combined.sort(key=lambda x: x[1])
    preds, targets = list(zip(*combined))

    plt.bar(range(1, 267), targets, alpha=0.7)
    plt.bar(range(1, 267), preds, alpha=0.7)

    plt.show()


if __name__ == '__main__':
    main()