from dataclasses import dataclass
from typing import Dict, List

import numpy
import torch

from omegaconf import DictConfig

# path context keys
FROM_TYPE = "from_type"
FROM_TOKEN = "from_token"
PATH_NODES = "path_nodes"
TO_TOKEN = "to_token"
TO_TYPE = "to_type"


@dataclass
class ContextPart:
    name: str
    to_id: Dict[str, int]
    parameters: DictConfig

@dataclass
class PathSample:
    contexts: numpy.ndarray
    paths: numpy.ndarray
    label: numpy.ndarray
    n_contexts: int
    n_paths: int


class PathBatch:
    def __init__(self, samples: List[PathSample]):
        samples = [s for s in samples if s is not None]
        self.contexts_per_label = [_s.n_contexts for _s in samples]
        self.paths_per_label = [_s.n_paths for _s in samples]

        torch_labels = numpy.hstack([_s.label for _s in samples])
        self.labels = torch.from_numpy(torch_labels)

        torch_contexts = numpy.hstack([_s.contexts for _s in samples])
        self.contexts = torch.from_numpy(torch_contexts)
        
        torch_paths = numpy.hstack([_s.paths for _s in samples])
        self.paths = torch.from_numpy(torch_paths)

    def __len__(self) -> int:
        return len(self.contexts_per_label)

    def pin_memory(self) -> "PathBatch":
        self.labels = self.labels.pin_memory()
        self.contexts = self.contexts.pin_memory()
        self.paths = self.paths.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.contexts = self.contexts.to(device)
        self.paths = self.paths.to(device)
