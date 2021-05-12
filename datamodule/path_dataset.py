from os.path import exists
from typing import Dict, List, Optional

import numpy as np
import json
import wordninja as ninja
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datamodule.data_classes import PathSample, FROM_TOKEN, PATH_NODES, TO_TOKEN, ContextPart
from utils.converting import strings_to_wrapped_numpy
from utils.vocabulary import Vocabulary


class PathDataset(Dataset):

    def __init__(self, data_file_path: str, config: DictConfig, vocabulary: Vocabulary, random_context: bool):
        if not exists(data_file_path):
            raise ValueError(f"Can't find file with data: {data_file_path}")
        self._data_file_path = data_file_path
        self._hyper_parameters = config.hyper_parameters
        self._random_context = random_context
        self._data = []
        with open(self._data_file_path, 'r') as f:
            self._data = json.load(f) 
        self._n_samples = len(self._data)

        self._target_vocabulary = vocabulary.label_to_id
        self._target_parameters = config.dataset.target

        self._token_vocabulary = vocabulary.token_to_id
        self._unit_parameters = config.dataset.unit

        self._path_parameters = config.dataset.path

    def __len__(self):
        return self._n_samples

    def __getitem__(self, index) -> Optional[PathSample]:
        raw_sample = self._data[index]
        str_label = raw_sample['name']
        int_paths = raw_sample['paths']
        str_context = raw_sample['body'] # 
        if str_label == "" or len(int_paths) == 0:
            return None

        
        n_contexts = min(len(str_context), self._hyper_parameters.max_lines)
        context_indexes = np.arange(n_contexts)
        str_context = str_context[:n_contexts]

        # convert string label to wrapped numpy array
        wrapped_label, recover = strings_to_wrapped_numpy(
            [str_label],
            self._target_vocabulary,
            self._target_parameters.is_split,
            self._target_parameters.max_length,
            self._target_parameters.use_SOS,
            self._target_parameters.use_EOS,
            " "
        )

        # convert each context to list of ints and then wrap into numpy array
        splitted_contexts, recover = strings_to_wrapped_numpy(
            str_context,
            self._token_vocabulary,
            self._unit_parameters.is_split,
            self._unit_parameters.max_length,
            self._unit_parameters.use_SOS,
            self._unit_parameters.use_EOS,
            " "
        )
        # choose random paths
        n_paths = min(len(int_paths), self._hyper_parameters.max_paths)
        context_indexes = np.arange(len(int_paths))
        if self._random_context:
            np.random.shuffle(context_indexes)
        context_indexes = context_indexes[:n_paths]

        splitted_paths = np.full((self._path_parameters.max_length, n_paths), self._path_parameters.path_PAD, dtype=np.int64)
        
        for i in range(n_paths):
            value = int_paths[context_indexes[-i]]
            length = min(len(value), self._path_parameters.max_length-1)
            splitted_paths[:length, i] = [_t for _t in value[:length]]
            splitted_paths[length, i] = self._path_parameters.path_EOS

        return PathSample(contexts=splitted_contexts, paths=splitted_paths, label=wrapped_label, n_contexts=n_contexts, n_paths = n_paths)