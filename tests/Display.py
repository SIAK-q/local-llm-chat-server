import os
import sys

import numpy

sys.path.append(os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), '..', '..'
    )
))

from typing import Any, Tuple
import math

from dlframe import DataSet,ListDataSet, Splitter, Model, Judger, WebManager
from sklearn import datasets
import numpy as np

class TestDataset(ListDataSet):
    def __init__(self, num,name:str) -> None:
        super().__init__(num)
        self.logger.print("I'm in range 0, {}".format(len(num)))
        self.name=name
        self.num = num

    def __len__(self) -> int:
        return len(self.num)
    def __getitem__(self, idx: int) -> Any:
        return self.num[idx]

class TrainTestDataset(ListDataSet):
    def __init__(self, item,name:str) -> None:
        super().__init__(item)
        self.name=name
        self.item = item
    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx: int) -> Any:
        return self.item[idx]

class TestSplitter(Splitter):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def split_(self, dataset: DataSet) -> Tuple[DataSet, DataSet]:
        f1 = [j for j in range(len(dataset))]
        import random
        random.shuffle(f1)
        train_size = math.floor(len(dataset) * self.ratio)
        tr = f1[:train_size]
        te = f1[train_size:]
        trainData = [dataset[m] for m in tr]
        testData = [dataset[n] for n in te]
        trainingSet = TrainTestDataset(trainData,dataset.name)
        testingSet = TrainTestDataset(testData,dataset.name)
        self.logger.print("split!")
        self.logger.print("training_len = {}".format(len(trainingSet)))
        return trainingSet, testingSet

    def split(self, dataset: DataSet) -> Tuple[DataSet, DataSet]:
        y=[dataset[i][1] for i in range(len(dataset))]
        y_lable=list(set(y))
        k=len(y_lable)
        trainData=list()
        testData=list()
        for i in range(k):
            index=[j for j,x in enumerate(y) if x==y_lable[i]]
            a_size=len(index)
            if a_size==0:
                continue
            train_size=math.floor(a_size*self.ratio)
            f1=[j for j in range(a_size)]
            import random
            random.shuffle(f1)

            tr=f1[:train_size]
            te=f1[train_size:]
            tr_set=[index[h] for h in tr]
            te_set=[index[g] for g in te]
            train_a=[dataset[m] for m in tr_set]
            test_a =[dataset[n] for n in te_set]
            trainData.extend(train_a)
            testData.extend(test_a)

        #trainingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio))]
        trainingSet = TrainTestDataset(trainData,dataset.name)
        #testingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio), len(dataset))]
        testingSet = TrainTestDataset(testData,dataset.name)

        self.logger.print("split!")
        self.logger.print("training_len = {}".format(len(trainingSet)))
        return trainingSet, testingSet

class TestModel(Model):
    def __init__(self, learning_rate,flag:int) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.flag=flag

    def train(self, trainDataset: DataSet) -> None:
        self.logger.print("trainging, lr = {}".format(self.learning_rate))
        if self.flag==1:
            self.logger.print("执行第一个模型")
        if self.flag==2:
            self.logger.print("执行第二个模型")

        return super().train(trainDataset)

    def test(self, testDataset: DataSet) -> Any:
        self.logger.print("testing")
        return testDataset

class TestJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, y_hat, test_dataset: DataSet) -> None:
        #self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))
        return super().judge(y_hat, test_dataset)


from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer


if __name__ == '__main__':
    WebManager().register_dataset(
        TestDataset([list(t) for t in zip(load_iris().data.tolist(),load_iris().target.tolist())],"鸢尾花"), '鸢尾花数据集'
    ).register_dataset(
        TestDataset([list(t) for t in zip(load_boston().data.tolist(),load_boston().target.tolist())],"波士顿"), '波士顿房价数据集'
    ).register_dataset(
        TestDataset([list(t) for t in zip(load_breast_cancer().data.tolist(),load_breast_cancer().target.tolist())],"乳腺癌"), '威斯康辛州乳腺癌数据集'
    ).register_splitter(
        TestSplitter(0.9), 'ratio:0.9'
    ).register_splitter(
        TestSplitter(0.8), 'ratio:0.8'
    ).register_splitter(
        TestSplitter(0.7), 'ratio:0.7'
    ).register_splitter(
        TestSplitter(0.6), 'ratio:0.6'
    ).register_splitter(
        TestSplitter(0.5), 'ratio:0.5'
    ).register_model(
        TestModel(1e-3,1),'第一个模型'
    ).register_model(
        TestModel(1e-3,2),'第二个模型'
    ).register_judger(
        TestJudger()
    ).start()