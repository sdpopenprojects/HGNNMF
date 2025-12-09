import os

os.environ["DGLBACKEND"] = "pytorch"

import torch
import torch.nn as nn
import torchmetrics

import dgl
from dgl.data import DGLDataset
import dgl.data.utils as dglUtils
from dgl.nn import GraphConv

import argparse
from tqdm import tqdm
import pandas as pd
import ast

from myProjectUtils import MyUtils, MyAccumulator


prediction_dir = "output/prediction/GCN2Defect/"
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


class MyDataset(DGLDataset):
    def __init__(
        self,
        projectName,
        projectVersion,
        isWPDP,
        force_reload=False,
        verbose=False,
    ):
        self.projectName = projectName
        self.projectVersion = projectVersion
        self.isWPDP = isWPDP

        super().__init__(
            name=projectName,
            raw_dir="./data/graph/cdn/raw/",
            save_dir="./data/graph/cdn/processed/",
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        nodesData = pd.read_csv(
            "{}/{}/nodes.csv".format(self.raw_path, self.projectVersion)
        )
        edgesData = pd.read_csv(
            "{}/{}/edges.csv".format(self.raw_path, self.projectVersion)
        )

        nodesData["node_metrics"] = nodesData["node_metrics"].apply(
            lambda x: [float(xxx) for xxx in ast.literal_eval(x)]
        )
        self.labels = torch.LongTensor(
            nodesData["node_label"].apply(lambda x: int(x)).to_list()
        )

        def getNormValues(values):
            trainData = torch.FloatTensor(values)

            min_vals, _ = torch.min(trainData, dim=0, keepdim=True)
            max_vals, _ = torch.max(trainData, dim=0, keepdim=True)

            return (trainData - min_vals) / (max_vals - min_vals + 0.00001)

        nodeMetrics = torch.Tensor(nodesData["node_metrics"].to_list())
        nodeMetrics = getNormValues(nodeMetrics)

        edgeSrc = torch.from_numpy(edgesData["edge_srcId"].to_numpy())
        edgeDst = torch.from_numpy(edgesData["edge_dstId"].to_numpy())

        self.graph = dgl.graph((edgeSrc, edgeDst), num_nodes=nodesData.shape[0])
        self.graph.ndata["feat"] = nodeMetrics
        self.graph.ndata["label"] = self.labels

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return self.graph.num_nodes()

    def download(self):
        pass

    def save(self):
        graph_path = os.path.join(self.save_path, self.projectVersion + "_graph.bin")
        dglUtils.save_graphs(graph_path, self.graph, {"labels": self.labels})

    def load(self):
        graph_path = os.path.join(self.save_path, self.projectVersion + "_graph.bin")
        self.graphs, label_dict = dglUtils.load_graphs(graph_path)
        self.labels = label_dict["labels"]


class MyGCN2Defect(nn.Module):
    def __init__(self, traditionalMetricsDim=18) -> None:
        super(MyGCN2Defect, self).__init__()
        self.convs = GraphConv(
            traditionalMetricsDim,
            traditionalMetricsDim,
            norm="both",
            weight=True,
            bias=True,
            activation=torch.relu,
            allow_zero_in_degree=True,
        )
        self.linear = nn.Linear(traditionalMetricsDim, traditionalMetricsDim)
        self.myclassify = nn.Linear(traditionalMetricsDim * 2, 2)

    def forward(self, graph, feat):
        with graph.local_scope():
            srcGraphTraMetricsF = graph.srcdata["feat"]
            graphTraMetricsF = self.convs(graph, srcGraphTraMetricsF)

            traMetricsF = self.linear(feat)
            graphTraMetricsF = torch.concat((traMetricsF, graphTraMetricsF), dim=1)
            return self.myclassify(torch.relu(graphTraMetricsF))


def evaluate(net, testDataLoader, device):
    with torch.no_grad():
        preds = []
        labels = []
        for input_nodes, output_nodes, blocks in testDataLoader:
            y = blocks[-1].dstdata["label"].to(device, dtype=torch.int64)

            blocks = [block.to(device) for block in blocks]
            y_hat = net(blocks[0], blocks[1].dstdata["feat"])

            y_hat = torch.argmax(y_hat, dim=1)

            preds.extend(y_hat.tolist())
            labels.extend(y.tolist())

        preds = torch.tensor(preds)
        labels = torch.tensor(labels)

        return (
            torchmetrics.functional.precision(preds, labels, "binary"),
            torchmetrics.functional.recall(preds, labels, "binary"),
            torchmetrics.functional.fbeta_score(preds, labels, "binary", beta=1.0),
            torchmetrics.functional.auroc(preds.to(torch.float32), labels, "binary"),
            torchmetrics.functional.matthews_corrcoef(preds, labels, "binary"),
        )


def trainEpoch(net, trainDataLoader, loss, weight, updater, device):
    metric = MyAccumulator(3)

    for input_nodes, output_nodes, blocks in trainDataLoader:
        # 最后一个块的输出节点的特征, 这里有batch个节点
        X = blocks[-1].dstdata["feat"]

        y = blocks[-1].dstdata["label"].to(device, dtype=torch.int64)

        blocks = [block.to(device) for block in blocks]
        y_hat = net(blocks[0], X)

        l = loss(y_hat, y.long(), weight=weight)
        updater.zero_grad()
        l.backward()
        updater.step()
        with torch.no_grad():
            metric.add(
                l.item() * X.shape[0],
                MyUtils.accuracy(y_hat, y),
                y.numel(),
            )
    return metric[0] / metric[2], metric[1] / metric[2]


def train(
    net,
    trainDataLoader,
    testDataLoader,
    num_epochs,
    updater,
    weight,
    device,
):
    net.apply(MyUtils.weight_init)
    net.to(device)
    net.train()

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = trainEpoch(
            net,
            trainDataLoader,
            torch.nn.functional.cross_entropy,
            weight,
            updater,
            device,
        )

        evam = evaluate(net, testDataLoader, device)

        if epoch % 64 == 0:
            print(
                "Epoch {}: train loss {:.4f}  train acc: {:.5f}  test acc {:.5f}  test P {:.5f}  test R {:.5f}\n".format(
                    epoch, train_loss, train_acc, evam[0], evam[1], evam[2]
                ),
                flush=True,
            )


def sdpProcess(trainProjectName, trainVersion, testProjectName, testVersion, device):

    lr, epochs, batchSize = (1e-4, 256, 64)

    trainDataSet, testDataSet = MyDataset(
        trainProjectName, trainVersion, trainProjectName == testProjectName
    ), MyDataset(testProjectName, testVersion, trainProjectName == testProjectName)

    trainDataLoader, testDataLoader = dgl.dataloading.DataLoader(
        trainDataSet.graph,
        trainDataSet.graph.nodes(),
        dgl.dataloading.MultiLayerFullNeighborSampler(1),
        batch_size=batchSize,
        shuffle=False,
        drop_last=False,
    ), dgl.dataloading.DataLoader(
        testDataSet.graph,
        testDataSet.graph.nodes(),
        dgl.dataloading.MultiLayerFullNeighborSampler(1),
        batch_size=batchSize,
        shuffle=False,
        drop_last=False,
    )

    model = MyGCN2Defect(trainDataSet.graph.ndata["feat"].shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    weight = MyUtils.class_weight(trainDataSet.labels.numpy(), device)
    train(model, trainDataLoader, testDataLoader, epochs, optimizer, weight, device)

    return evaluate(model, testDataLoader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trainProjectName",
        choices=["ant", "activemq", "lucene", "jruby", "hbase", "hive"],
    )
    parser.add_argument("trainVersion", type=str)
    parser.add_argument(
        "testProjectName",
        type=str,
        choices=["ant", "activemq", "lucene", "jruby", "hbase", "hive"],
    )
    parser.add_argument("testVersion", type=str)
    parser.add_argument("runTimes", type=int)
    args = parser.parse_args()

    trainProjectName, testProjectName, trainVersion, testVersion, runTimes = (
        args.trainProjectName,
        args.testProjectName,
        args.trainVersion,
        args.testVersion,
        args.runTimes,
    )

    MyUtils.commonMain(
        runTimes,
        trainProjectName,
        trainVersion,
        testProjectName,
        testVersion,
        prediction_dir,
        sdpProcess,
    )
