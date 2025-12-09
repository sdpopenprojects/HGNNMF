import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchmetrics
import pandas as pd
from tqdm import tqdm


class MyUtils(object):
    @staticmethod
    def try_gpu(i=0):
        if i < 0:
            return torch.device("cpu")
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f"cuda:{i}")
        return torch.device("cpu")

    @staticmethod
    def class_weight(y, device):
        neg, pos = np.bincount(y.tolist())
        weight = [(1.0 / neg) * (pos + neg) / 2.0, (1.0 / pos) * (pos + neg) / 2.0]
        return torch.FloatTensor(weight).to(device)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def train_epoch(net, train_iter, loss, weight, updater, device):
        if isinstance(net, torch.nn.Module):
            net.train()

        metric = MyAccumulator(3)

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y, weight=weight)
            updater.zero_grad()
            l.backward()
            updater.step()
            with torch.no_grad():
                metric.add(
                    l * X.shape[0],
                    MyUtils.accuracy(y_hat, y),
                    y.numel(),
                )
        return metric[0] / metric[2], metric[1] / metric[2]

    @staticmethod
    def commonMain(
        runTimes,
        trainProjectName,
        trainProjectVersion,
        testProjectName,
        testProjectVersion,
        prediction_dir,
        sdpProcess,
    ):
        predictionDF = pd.DataFrame(
            {
                "RunTimes": [],
                "TrainProjectName": [],
                "TrainProjectVersion": [],
                "TestProjectName": [],
                "TestProjectVersion": [],
                "P": [],
                "R": [],
                "F1": [],
                "AUC": [],
                "MCC": [],
            },
        )
        predictionDF = predictionDF.astype(
            {
                "RunTimes": int,
                "TrainProjectName": str,
                "TrainProjectVersion": str,
                "TestProjectName": str,
                "TestProjectVersion": str,
                "P": float,
                "R": float,
                "F1": float,
                "AUC": float,
                "MCC": float,
            }
        )

        for i in tqdm(range(runTimes)):
            print(
                "RunTimes {} : {}--{}-----{}--{}\n".format(
                    i + 1,
                    trainProjectName,
                    trainProjectVersion,
                    testProjectName,
                    testProjectVersion,
                ),
                flush=True,
            )

            evam = sdpProcess(
                trainProjectName,
                trainProjectVersion,
                testProjectName,
                testProjectVersion,
                MyUtils.try_gpu(),
            )

            predictionDF = pd.concat(
                [
                    predictionDF,
                    pd.DataFrame(
                        {
                            "RunTimes": [i + 1],
                            "TrainProjectName": [trainProjectName],
                            "TrainProjectVersion": [trainProjectVersion],
                            "TestProjectName": [testProjectName],
                            "TestProjectVersion": [testProjectVersion],
                            "P": [evam[0].item()],
                            "R": [evam[1].item()],
                            "F1": [evam[2].item()],
                            "AUC": [evam[3].item()],
                            "MCC": [evam[4].item()],
                        },
                    ),
                ],
                ignore_index=True,
            )

            MyUtils.printMetrics(evam)

        predictionDF.to_csv(
            prediction_dir
            + "{}_{}-{}_{}.csv".format(
                trainProjectName,
                trainProjectVersion,
                testProjectName,
                testProjectVersion,
            ),
            index=None,
        )

    @staticmethod
    def printMetrics(metrics):
        print(
            "===========P=============\n{}\n===========R=============\n{}\n===========F1=============\n{}\n===========AUC=============\n{}\n===========MCC=============\n{}".format(
                metrics[0].item(),
                metrics[1].item(),
                metrics[2].item(),
                metrics[3].item(),
                metrics[4].item(),
            ),
            flush=True,
        )

    @staticmethod
    def evaluate(net, data_iter, device, computeF1=False):
        if isinstance(net, torch.nn.Module):
            net.eval()
        net.to(device)

        labels, preds = [], []

        with torch.no_grad():
            for X, y in data_iter:
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                preds.append(y_hat.argmax(1))
                labels.append(y)

            if computeF1:
                return (
                    torchmetrics.functional.precision(
                        torch.cat(preds), torch.cat(labels), "binary"
                    ),
                    torchmetrics.functional.recall(
                        torch.cat(preds), torch.cat(labels), "binary"
                    ),
                    torchmetrics.functional.fbeta_score(
                        torch.cat(preds), torch.cat(labels), "binary", beta=1.0
                    ),
                    torchmetrics.functional.auroc(
                        torch.cat(preds).float(), torch.cat(labels), "binary"
                    ),
                    torchmetrics.functional.matthews_corrcoef(
                        torch.cat(preds).float(), torch.cat(labels), "binary"
                    ),
                )
            else:
                return (
                    torchmetrics.functional.accuracy(
                        torch.cat(preds), torch.cat(labels), "binary"
                    ),
                    torchmetrics.functional.precision(
                        torch.cat(preds), torch.cat(labels), "binary"
                    ),
                    torchmetrics.functional.recall(
                        torch.cat(preds), torch.cat(labels), "binary"
                    ),
                )

    @staticmethod
    def train(
        net,
        train_iter,
        test_iter,
        num_epochs,
        updater,
        device,
        printEpochs=64,
    ):
        net.apply(MyUtils.weight_init)
        weight = MyUtils.class_weight(train_iter.dataset.labels.numpy(), device)
        net.to(device)
        for epoch in range(num_epochs):
            train_loss, train_acc = MyUtils.train_epoch(
                net, train_iter, F.cross_entropy, weight, updater, device
            )
            evam = MyUtils.evaluate(net, test_iter, device)
            if epoch % printEpochs == 0:
                print(
                    "Epoch {}: train loss {:.4f}  train acc: {:.5f}  test acc {:.5f}  test P {:.5f}  test R {:.5f}\n".format(
                        epoch, train_loss, train_acc, evam[0], evam[1], evam[2]
                    ),
                    flush=True,
                )

    @staticmethod
    def accuracy(y_hat, y):
        """Compute the number of correct predictions.
        Defined in :numref:`sec_softmax_scratch`"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = argmax(y_hat, axis=1)
        cmp = astype(y_hat, y.dtype) == y
        return float(reduce_sum(astype(cmp, y.dtype)))

    @staticmethod
    def tokenizerEncode(tokenizer, item):
        input_ids = tokenizer.encode(
            text=item,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding=False,
        )

        num_pads = 512 - len(input_ids)
        padded_input = input_ids + [tokenizer.pad_token_id] * num_pads
        atten_mask = [1] * len(input_ids) + [0] * num_pads

        return [padded_input, atten_mask]


class MyAccumulator(object):
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
