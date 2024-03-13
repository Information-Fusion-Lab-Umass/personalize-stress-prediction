import gzip
import pickle
from abc import ABC, abstractmethod

from arpreprocessing.dataset import Dataset
from arpreprocessing.subject import Subject


class Preprocessor(ABC):
    def __init__(self, logger, path, name, channels_names, get_sampling_fn, subject_cls=Subject):
        self._logger = logger
        self._path = path
        self._name = name
        self.subjects = [subject_cls(self._logger, self._path, id_, channels_names, get_sampling_fn) for id_ in
                         self.get_subjects_ids()]

    def get_dataset(self):
        return Dataset(self._name, self, self._logger)

    @abstractmethod
    def get_subjects_ids(self):
        raise NotImplementedError()


def load_gzipped_pkl(path):
    with gzip.open(path + ".pickle.gz", 'rb') as f:
        load = pickle.load(f)
        return load
