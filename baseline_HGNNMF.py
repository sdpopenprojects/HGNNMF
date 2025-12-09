import os

os.environ["DGLBACKEND"] = "pytorch"

import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim

import dgl
from dgl.data import DGLDataset
import dgl.data.utils as dglUtils
from dgl.dataloading import GraphDataLoader
import dgl.data
from dgl.nn import GATv2Conv, GlobalAttentionPooling, GraphConv

import argparse
from tqdm import tqdm
import pandas as pd
import ast

from myProjectUtils import MyUtils, MyAccumulator

prediction_dir = "output/prediction/HGNNMF_DEBUG/"
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


class MyCDNDataset(DGLDataset):
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


class MyCFGDataset(DGLDataset):
    def __init__(
        self,
        trainProjectName,
        trainVersion,
        testProjectName,
        testVersion,
        isTraining,
    ):
        self.instruction = {
            # core statements
            "soot.jimple.internal.JNopStmt": 1,
            "soot.jimple.internal.JIdentityStmt": 2,
            "soot.jimple.internal.JAssignStmt": 3,
            ## intraprocedural control
            "soot.jimple.internal.JIfStmt": 4,
            "soot.jimple.internal.JGotoStmt": 5,
            "soot.jimple.internal.JTableSwitchStmt": 6,
            "soot.jimple.internal.JLookupSwitchStmt": 7,
            ## interprocedural control
            "soot.jimple.internal.JInvokeStmt": 8,
            "soot.jimple.internal.JReturnStmt": 9,
            "soot.jimple.internal.JReturnVoidStmt": 10,
            ## monitor statements
            "soot.jimple.internal.JEnterMonitorStmt": 11,
            "soot.jimple.internal.JExitMonitorStmt": 12,
            ## others
            "soot.jimple.internal.JThrowStmt": 13,
            "soot.jimple.internal.JRetStmt": 14,
        }
        self.instruction_enc = F.one_hot(torch.arange(len(self.instruction) + 1))

        self.project, self.version = (
            (trainProjectName, trainVersion)
            if isTraining
            else (testProjectName, testVersion)
        )
        super().__init__(name="CFG")

    def process(self):
        nodes = pd.read_csv(
            "data/graph/cfg/{}/{}/nodes.csv".format(self.project, self.version)
        )
        edges = pd.read_csv(
            "data/graph/cfg/{}/{}/edges.csv".format(self.project, self.version)
        )
        properties = pd.read_csv(
            "data/graph/cfg/{}/{}/properties.csv".format(self.project, self.version)
        )

        self.graphs = []
        for _ in range(properties["cfg_id"].max() + 1):
            g = dgl.DGLGraph(([0], [0]))
            g.ndata["instruction_type"] = self.instruction_enc[0].reshape(1, -1).float()
            self.graphs.append(g)
        self.labels = [0] * (properties["cfg_id"].max() + 1)

        self.instructions = list(set(nodes["instruction"].values.tolist()))

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row["cfg_id"]] = 1 if row["label"] >= 1 else 0
            num_nodes_dict[row["cfg_id"]] = row["num_nodes"]

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby("cfg_id")
        nodes_group = nodes.groupby("cfg_id")

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            nodes_of_id = nodes_group.get_group(graph_id).sort_values(by="node_id")
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata["in_degree"] = g.in_degrees().view(-1, 1).float()
            g.ndata["out_degree"] = g.out_degrees().view(-1, 1).float()
            g.ndata["instruction_index"] = torch.tensor(
                list(
                    map(
                        lambda x: self.instructions.index(x),
                        nodes_of_id["instruction"].values.tolist(),
                    )
                )
            ).long()
            g.ndata["instruction_type"] = torch.stack(
                list(
                    map(
                        lambda x: (
                            self.instruction_enc[self.instruction[x]]
                            if self.instruction[x]
                            else self.instruction_enc[0]
                        ),
                        nodes_of_id["instruction_type"].values.tolist(),
                    )
                )
            ).float()
            self.graphs[graph_id] = g
            self.labels[graph_id] = label

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class MyGATv2(nn.Module):

    def __init__(self, h_dim, num_heads, feat_dropout, attn_dropout, layers) -> None:
        super(MyGATv2, self).__init__()

        self.convsN = nn.Sequential(
            GATv2Conv(
                15,
                h_dim,
                num_heads=num_heads,
                feat_drop=feat_dropout,
                attn_drop=attn_dropout,
                residual=True,
                activation=F.relu,
                allow_zero_in_degree=True,
            )
        )

        for i in range(1, layers):
            self.convsN.add_module(
                str(i),
                GATv2Conv(
                    h_dim * num_heads,
                    h_dim,
                    num_heads=num_heads,
                    feat_drop=feat_dropout,
                    attn_drop=attn_dropout,
                    residual=True,
                    activation=F.relu,
                    allow_zero_in_degree=True,
                ),
            )

        self.gPool = GlobalAttentionPooling(nn.Linear(h_dim * num_heads, 1))
        self.dropout = nn.Dropout(feat_dropout)

        self.linear = nn.Linear(h_dim * num_heads, 2 * h_dim)

    def forward(self, graph, feat):
        hN = feat
        hg = None
        for i in range(len(self.convsN)):
            hN = F.relu(self.convsN[i](graph, hN).flatten(1))

            with graph.local_scope():
                graph.ndata["h"] = hN
                newhN = dgl.softmax_nodes(graph, "h")

                if hg == None:
                    hg = self.gPool(graph, newhN)
                else:
                    hg = hg + self.gPool(graph, newhN)

        return self.linear(self.dropout(hg))


class MyGCN2Defect(nn.Module):
    def __init__(self, traditionalMetricsDim=18, hDim=32) -> None:
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
        self.linear = nn.Linear(traditionalMetricsDim + 2 * hDim, hDim)

    def forward(self, graph, feat):
        with graph.local_scope():
            srcGraphTraMetricsF = graph.srcdata["feat"]
            graphTraMetricsF = self.convs(graph, srcGraphTraMetricsF)

            traMetricsF = self.linear(feat)
            return torch.concat((traMetricsF, graphTraMetricsF), dim=1)


class MyHGNNMF(nn.Module):
    def __init__(
        self,
        h_dim,
        num_heads,
        feat_dropout,
        attn_dropout,
        layers,
        traditionalMetricsDim=18,
    ) -> None:
        super(MyHGNNMF, self).__init__()
        self.subgraphEncoder = MyGATv2(
            h_dim, num_heads, feat_dropout, attn_dropout, layers
        )
        self.globalEncoder = MyGCN2Defect(traditionalMetricsDim, h_dim)

        self.myclassify = nn.Linear(h_dim + traditionalMetricsDim, 2)

    def __getFeature(self, batchg):
        return torch.cat(
            (batchg.ndata["instruction_type"],),
            1,
        )

    def forward(self, subGraphs, g, traFeat):
        with g.local_scope():
            globalNodeFeat = []

            for subgraph in subGraphs:
                feat = self.__getFeature(subgraph)
                h = self.subgraphEncoder(subgraph, feat)
                globalNodeFeat.append(h)
            globalNodeFeat = torch.stack(globalNodeFeat).squeeze(1)
            #


            # 对外层图进行图卷积
            gH = self.globalEncoder(g, torch.concat((globalNodeFeat, traFeat), dim=1))
            return self.myclassify(gH)


def evaluate(net, testDataLoader, testCFGDataset, device):
    with torch.no_grad():
        preds = []
        labels = []
        for input_nodes, output_nodes, blocks in testDataLoader:
            X = blocks[-1].dstdata["feat"].to(device)

            subGraphs = []
            for tmpNode in output_nodes:
                tmpg = testCFGDataset.graphs[tmpNode]
                subGraphs.append(tmpg.to(device))

            y = blocks[-1].dstdata["label"].to(device, dtype=torch.int64)

            blocks = [block.to(device) for block in blocks]
            y_hat = net(subGraphs, blocks[0], X)

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


def trainEpoch(net, trainDataLoader, loss, weight, updater, trainCFGDataset, device):
    metric = MyAccumulator(3)

    for input_nodes, output_nodes, blocks in trainDataLoader:
        # 最后一个块的输出节点的特征, 这里有batch个节点
        X = blocks[-1].dstdata["feat"].to(device)

        subGraphs = []
        for tmpNode in output_nodes:
            tmpg = trainCFGDataset.graphs[tmpNode]
            subGraphs.append(tmpg.to(device))

        y = blocks[-1].dstdata["label"].to(device, dtype=torch.int64)

        blocks = [block.to(device) for block in blocks]
        y_hat = net(subGraphs, blocks[0], X)

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
    trainCFGDataset,
    testCFGDataset,
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
            trainCFGDataset,
            device,
        )

        evam = evaluate(net, testDataLoader, testCFGDataset, device)

        if epoch % 64 == 0:
            print(
                "Epoch {}: train loss {:.4f}  train acc: {:.5f}  test acc {:.5f}  test P {:.5f}  test R {:.5f}\n".format(
                    epoch, train_loss, train_acc, evam[0], evam[1], evam[2]
                ),
                flush=True,
            )


def sdpProcess(trainProjectName, trainVersion, testProjectName, testVersion, device):

    lr, epochs, batchSize = (1e-3, 500, 64)  # 1e-4, 200, 32   1e-4, 16, 16
    hDim, numHeads, layers, dropout = 32, 5, 4, 0.3  # 30, 5, 2, 0.2  32, 3, 2, 0.3

    trainCDNDataSet, testCDNDataSet = MyCDNDataset(
        trainProjectName, trainVersion, trainProjectName == testProjectName
    ), MyCDNDataset(testProjectName, testVersion, trainProjectName == testProjectName)

    trainCFGDataset, testCFGDataset = MyCFGDataset(
        trainProjectName, trainVersion, testProjectName, testVersion, True
    ), MyCFGDataset(testProjectName, testVersion, testProjectName, testVersion, False)

    trainDataLoader, testDataLoader = dgl.dataloading.DataLoader(
        trainCDNDataSet.graph,
        trainCDNDataSet.graph.nodes(),
        dgl.dataloading.MultiLayerFullNeighborSampler(1),
        batch_size=batchSize,
        shuffle=False,
        drop_last=False,
    ), dgl.dataloading.DataLoader(
        testCDNDataSet.graph,
        testCDNDataSet.graph.nodes(),
        dgl.dataloading.MultiLayerFullNeighborSampler(1),
        batch_size=batchSize,
        shuffle=False,
        drop_last=False,
    )

    model = MyHGNNMF(
        hDim,
        numHeads,
        dropout,
        dropout,
        layers,
        traditionalMetricsDim=trainCDNDataSet.graph.ndata["feat"].shape[1],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    weight = MyUtils.class_weight(trainCDNDataSet.labels.numpy(), device)
    train(
        model,
        trainDataLoader,
        testDataLoader,
        epochs,
        optimizer,
        weight,
        trainCFGDataset,
        testCFGDataset,
        device,
    )

    return evaluate(model, testDataLoader, testCFGDataset, device)


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
