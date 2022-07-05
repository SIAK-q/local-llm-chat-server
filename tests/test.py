import os
import sys
from typing import Tuple
sys.path.append(os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), '..', '..'
    )
))
import sklearn
import math
from traitlets import Any
from dlframe import DataSet, Splitter, Model, Judger, WebManager

# 数据集
class TestDataset(DataSet):
    def __init__(self, num) -> None:
        super().__init__()
        self.num = range(num)
        self.logger.print("I'm in range 0, {}".format(num))

    def __len__(self) -> int:
        return len(self.num)

    def __getitem__(self, idx: int) -> Any:
        return self.num[idx]

class TrainTestDataset(DataSet):
    def __init__(self, item) -> None:
        super().__init__()
        self.item = item

    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx: int) -> Any:
        return self.item[idx]

# 数据集切分器
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
        self.logger.print("training_len = {}".format([trainingSet[i] for i in range(len(trainingSet))]))
        return trainingSet, testingSet

# 模型
class TestModel(Model):
    def __init__(self, learning_rate, flag) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.flag = flag

    def train(self, trainDataset: DataSet) -> None:
        if self.flag == 1:
            self.logger.print("Model 1")
        elif self.flag == 2:
             self.logger.print("Model 2")
        self.logger.print("trainging, lr = {}".format(self.learning_rate))
        return super().train(trainDataset)

    def test(self, testDataset: DataSet) -> Any:
        self.logger.print("testing")
        return testDataset


# 结果判别器
class TestJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, y_hat, test_dataset: DataSet) -> None:
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))
        return super().judge(y_hat, test_dataset)

if __name__ == '__main__':
    # 注册与运行
    WebManager().register_dataset(
        TestDataset(10), '10_nums'
    ).register_dataset(
        TestDataset(20), '20_nums'
    ).register_splitter(
        TestSplitter(0.8), 'shdfuishfratio:0.8'
    ).register_splitter(
        TestSplitter(0.5), 'ratio:0.5'
    ).register_model(
        TestModel('1e-3',1),'Model 1'
    ).register_model(
        TestModel('1e-3',2),'Model 2'
    ).register_judger(
        TestJudger()
    ).start()