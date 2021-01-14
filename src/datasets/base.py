from ..utils import DATA_ROOT_FOLDER

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):

        self.args = args
        self.split = args.split

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    def _get_rawdata_root_path(self):
        return Path(DATA_ROOT_FOLDER)

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return Path("../../")

    @abstractmethod
    def _get_preprocessed_folder_path(self):
        # preprocessed_root = self._get_preprocessed_root_path()
        #
        # return preprocessed_root.joinpath(folder_name)
        pass

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl') # data/squad/dataset.pkl