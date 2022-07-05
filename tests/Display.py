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
    def __init__(self, num) -> None:
        super().__init__(num)
        self.logger.print("I'm in range 0, {}".format(len(num)))
        self.num = num

    def __len__(self) -> int:
        return len(self.num)
    def __getitem__(self, idx: int) -> Any:
        return self.num[idx]

class TrainTestDataset(ListDataSet):
    def __init__(self, item) -> None:
        super().__init__(item)
        self.item = item
    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx: int) -> Any:
        return self.item[idx]

#class TestDataset(DataSet):
 #   def __init__(self, num:int) -> None:
  #      super().__init__()
  #      self.num = range(num)
  #      self.logger.print("I'm in range 0, {}".format(num))
#
  #  def __len__(self) -> int:
  #      return len(self.num)
#    def __getitem__(self, idx: int) -> Any:
#        return self.num[idx]

#class TrainTestDataset(DataSet):
#    def __init__(self, item) -> None:
#        super().__init__()
#        self.item = item

#    def __len__(self) -> int:
#        return len(self.item)

#    def __getitem__(self, idx: int) -> Any:
#        return self.item[idx]

class TestSplitter(Splitter):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset: DataSet) -> Tuple[DataSet, DataSet]:
        trainingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio))]
        trainingSet = TrainTestDataset(trainingSet)

        testingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio), len(dataset))]
        testingSet = TrainTestDataset(testingSet)

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
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))
        return super().judge(y_hat, test_dataset)


from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer


if __name__ == '__main__':
    WebManager().register_dataset(
        TestDataset(load_iris().data.tolist()), '鸢尾花数据集'
    ).register_dataset(
        TestDataset(load_boston().data.tolist()), '波士顿房价数据集'
    ).register_dataset(
        TestDataset(load_breast_cancer().data.tolist()), '威斯康辛州乳腺癌数据集'
    ).register_splitter(
        TestSplitter(0.8), 'ratio:0.8'
    ).register_splitter(
        TestSplitter(0.5), 'ratio:0.5'
    ).register_model(
        TestModel(1e-3,1),'第一个模型'
    ).register_model(
        TestModel(1e-3,2),'第二个模型'
    ).register_judger(
        TestJudger()
    ).start()