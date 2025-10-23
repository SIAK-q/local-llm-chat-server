from tests.TestCmdManager import DebugDataset, DebugModel, DebugJudger
from dlframe.webmanager import WebManager
from dlframe.splitter import DirectSplitter

if __name__ == '__main__':
    WebManager().register_dataset(
        DebugDataset(10), '10_ints'
    ).register_splitter(
        DirectSplitter(), 'split-direct'
    ).register_model(
        DebugModel()
    ).register_judger(
        DebugJudger()
    ).start(host='127.0.0.1', port=8765)
