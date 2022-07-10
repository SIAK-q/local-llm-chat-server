from ast import List
from dlframe.webitem import WebItem

from typing import Any

class DataSet(WebItem):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Any:
        pass

    def __getcontent__(self)-> List:
        return []

class ListDataSet(DataSet):
    def __init__(self, innerList) -> None:
        super().__init__()
        self.innerList = innerList

    def __len__(self) -> int:
        return len(self.innerList)

    def __getitem__(self, idx: int) -> Any:
        return self.innerList[idx]