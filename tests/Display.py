import os
import sys

import numpy
from sklearn import metrics
from sklearn.metrics import accuracy_score,jaccard_score,hinge_loss,hamming_loss,cohen_kappa_score,calinski_harabasz_score, fowlkes_mallows_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor as XGBR


sys.path.append(os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), '..', '..'
    )
))

from typing import Any, Tuple
import math

from dlframe import DataSet,ListDataSet, Splitter, Model, Judger, WebManager
from sklearn import datasets, svm
import numpy as np

from algorithm import 决策树,梯度增强,概率图,贝叶斯分类器
from algorithm.贝叶斯分类器 import NaiveBayes
from algorithm.决策树 import DecisionTree
from algorithm.梯度增强 import GradientBoostingRegressor
from algorithm import XGboost

params = {
    'eta': 0.02,  #lr
    'num_class':3,
    'max_depth': 6, 
    'min_child_weight':3,#最小叶子节点样本权重和
    'gamma':0, #指定节点分裂所需的最小损失函数下降值。
    'subsample': 0.7,  #控制对于每棵树，随机采样的比例
    'colsample_bytree': 0.3,  #用来控制每棵随机采样的列数的占比 (每一列是一个特征)。
    'lambda':2,
    'objective': 'multi:softmax', 
    'eval_metric': 'mlogloss', 
    'silent': True, 
    'nthread': -1
}

class TestDataset(ListDataSet):
    def __init__(self, num,name:str) -> None:
        super().__init__(num)
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

    def split(self, dataset: DataSet) -> Tuple[DataSet, DataSet]:
        if dataset.name=="鸢尾花" or dataset.name=="乳腺癌":
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
        else:
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

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.cluster import KMeans


class TestModel(Model):
    def __init__(self, learning_rate,name:str) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.name=name

    def train(self, trainDataset: DataSet) -> None:
        train_X = [trainDataset[i][0] for i in range(len(trainDataset))]
        train_Y = [trainDataset[i][1] for i in range(len(trainDataset))]
        self.logger.print("trainging, lr = {}".format(self.learning_rate))
        if self.name=="决策树":
            self.jueceshuModel = DecisionTreeClassifier()
            self.jueceshuModel.fit(train_X,train_Y)
            self.logger.print("执行决策树算法")
        if self.name=="贝叶斯分类器":
            self.beyesiModel=GaussianNB()
            self.beyesiModel.fit(train_X, train_Y)
            self.logger.print("执行贝叶斯分类器算法")
        if self.name=="梯度增强":
            self.tiduzengqiangModel=GradientBoostingClassifier()
            self.tiduzengqiangModel.fit(train_X,train_Y)
            self.logger.print("执行梯度增强算法")
        if self.name=="线性回归":
            self.LrModel=LinearRegression()
            self.LrModel.fit(train_X,train_Y)
            self.logger.print("执行线性回归算法")
        if self.name=="k-近邻":
            self.knn = KNeighborsClassifier() 
            self.knn.fit(train_X,train_Y) 
            self.logger.print("执行k-近邻算法")
        if self.name=="XGboost":
            # self.xgb_model = XGboost()
            # self.xgb_model.fit(train_X,train_Y)
            self.reg = XGBR(n_estimators=100)
            self.reg.fit(train_X,train_Y)
            self.logger.print("执行XGboost算法")
        if self.name=="SVM":
            self.classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo') 
            self.classifier.fit(train_X,train_Y) 
            self.logger.print("执行SVM算法")
        if self.name=="随机森林":
            self.suijisenlinModel=RandomForestClassifier()
            self.suijisenlinModel.fit(train_X,train_Y)
            self.logger.print("执行随机森林算法")
        if self.name == "逻辑回归":
            self.luojihuiguiModel=LogisticRegression()
            self.luojihuiguiModel.fit(train_X,train_Y)
            self.logger.print("执行逻辑回归算法")
        if self.name == "K-means聚类":
            self.kmeansModel=KMeans()
            self.kmeansModel.fit(train_X,train_Y)
            self.logger.print("执行K-means聚类算法")

        return super().train(trainDataset)

    def test(self, testDataset: DataSet) -> Any:
        test_X = [testDataset[i][0] for i in range(len(testDataset))]
        test_ans = [testDataset[i][1] for i in range(len(testDataset))]
        if self.name=="决策树":
            test_Y=self.jueceshuModel.predict(test_X)
        if self.name=="贝叶斯分类器":
            test_Y = self.beyesiModel.predict(test_X)
        if self.name=="梯度增强":
            test_Y=self.tiduzengqiangModel.predict(test_X)
        if self.name=="线性回归":
            test_Y=self.LrModel.predict(test_X)
        if self.name=="k-近邻":
            test_Y = self.knn.predict(test_X)
        if self.name=="XGboost":          
            test_Y=self.reg.predict(test_X)
        if self.name=="SVM":
            test_Y = self.classifier.predict(test_X)
        if self.name=="随机森林":
            test_Y=self.suijisenlinModel.predict(test_X)
        if self.name=="逻辑回归":
            test_Y=self.luojihuiguiModel.predict(test_X)
        if self.name=="K-means聚类":
            test_Y=self.kmeansModel.predict(test_X)

        self.logger.print("testing")
        self.logger.print("test_Y={}".format([test_Y[i] for i in range(len(test_Y))]))
        return test_Y

class TestJudger(Judger):
    def __init__(self,name:str) -> None:
        super().__init__()
        self.name=name

    def judge(self, y_hat, test_dataset: DataSet) -> None:
        self.logger.print("gt = {}".format([test_dataset[i][1] for i in range(len(test_dataset))]))
        test_X = [test_dataset[i][0] for i in range(len(test_dataset))]
        test_Y = [test_dataset[i][1] for i in range(len(test_dataset))]
        # self.logger.print("test_Y={}".format([test_Y[i] for i in range(len(test_Y))]))
        # self.logger.print("Y_hat={}".format([y_hat[i] for i in range(len(y_hat))]))
        if self.name == "分类判别":
            self.logger.print("执行分类判别")
            self.accuancy_score=accuracy_score(test_Y,y_hat)
            self.logger.print('准确率为：%f%%' % ((self.accuancy_score)*100))
            self.kappa=cohen_kappa_score(test_Y,y_hat)
            self.logger.print('分类kappa系数值为：%f' %(self.kappa))
            self.ham_distance=hamming_loss(test_Y,y_hat)
            self.logger.print('hamming距离值为：%f' % (self.ham_distance))


            pass
        if self.name=="聚类判别":
            self.logger.print("执行聚类判别")
            self.fmi_score = fowlkes_mallows_score(test_Y,y_hat)
            self.logger.print('聚类FMI评价分值为：%f' %(self.fmi_score))
            self.ch_score = calinski_harabasz_score(test_X,y_hat)
            self.logger.print('聚类calinski_harabaz指数为：%f'%(self.ch_score))

        if self.name=="回归判别":
            self.logger.print("执行回归判别")
            self.mse=mean_squared_error(test_Y,y_hat)
            self.rmse=self.mse**0.5
            self.mae=mean_absolute_error(test_Y,y_hat)
            self.mape = metrics.mean_absolute_percentage_error(test_Y, y_hat)
            self.r2=r2_score(test_Y,y_hat)
            self.logger.print("均方误差={}".format(self.mse,1))
            self.logger.print("均方根误差={}".format(self.rmse,1))
            self.logger.print("平均绝对误差={}".format(self.mae,1))
            self.logger.print("平均绝对百分比误差={}".format(self.mape,1))
            # mape 输出不是 [0, 100] 范围内的百分比，值 100 并不意味着 100%，而是 1e2。此外，当y_true 很小(特定于指标)或当abs(y_true - y_pred) 很大(这对于大多数回归指标很常见)时，输出可以任意高。
            self.logger.print("决定系数={}".format(self.r2,1))

        return super().judge(y_hat, test_dataset)


from sklearn.datasets import load_iris,load_boston,load_breast_cancer,load_diabetes,load_linnerud

if __name__ == '__main__':
    WebManager().register_dataset(
        TestDataset([list(t) for t in zip(load_iris().data.tolist(),load_iris().target.tolist())],"鸢尾花"), '鸢尾花数据集'
    ).register_dataset(
        TestDataset([list(t) for t in zip(load_breast_cancer().data.tolist(),load_breast_cancer().target.tolist())],"乳腺癌"), '威斯康辛州乳腺癌数据集'
    ).register_dataset(
        TestDataset([list(t) for t in zip(load_boston().data.tolist(),load_boston().target.tolist())],"波士顿"), '波士顿房价数据集'
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
        TestModel(1e-3,"决策树"),'决策树'
    ).register_model(
        TestModel(1e-3,"贝叶斯分类器"),'贝叶斯分类器'
    ).register_model(
        TestModel(1e-3,"梯度增强"),'梯度增强'
    ).register_model(
        TestModel(1e-3,"k-近邻"),'k-近邻'
    ).register_model(
        TestModel(1e-3,"SVM"),'SVM'
    ).register_model(
        TestModel(1e-3,"随机森林"),'随机森林'
    ).register_model(
        TestModel(1e-3,"逻辑回归"),'逻辑回归'
    ).register_model(
        TestModel(1e-3,"线性回归"),'线性回归'
    ).register_model(
        TestModel(1e-3,"XGboost"),'XGboost'
    ).register_model(
        TestModel(1e-3,"K-means聚类"),'K-means聚类'
    ).register_judger(
        TestJudger("分类判别"),'分类判别'
    ).register_judger(
        TestJudger("聚类判别"),'聚类判别'
    ).register_judger(
        TestJudger("回归判别"),'回归判别'
    ).start()