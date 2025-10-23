from ast import List
from typing import Any, Dict
from dlframe.webitem import WebItem
from dlframe.dataset import DataSet

class Model(WebItem):
    def __init__(self) -> None:
        super().__init__()

    def train(self, trainDataset: DataSet, param: Dict=None) -> None:
        pass

    def test(self, testDataset: DataSet, param: Dict=None) -> Any:
        pass

    def __getparams__(self)-> Dict:
        return {}
