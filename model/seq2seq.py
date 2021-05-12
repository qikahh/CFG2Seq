from typing import Dict, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from datamodule import PathBatch
from model.modules import SeqEncoder, SeqDecoder
from utils.metrics import PredictionStatistic
from utils.training import configure_optimizers_alon
from utils.vocabulary import Vocabulary, SOS, PAD, UNK, EOS


class Seq2Seq(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary
        self.save_hyperparameters()

        self._metric_skip_tokens = [
            vocabulary.label_to_id[i] for i in [PAD, UNK, EOS, SOS] if i in vocabulary.label_to_id
        ]
        self._label_pad_id = vocabulary.label_to_id[PAD]

        if SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")
        
        self.token_embedding = nn.Embedding(len(self._vocabulary.label_to_id), config.encoder.embedding_size, padding_idx=self._vocabulary.label_to_id[PAD])
        nn.init.xavier_normal_(self.token_embedding.weight)
        
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.decoder.projection_layer.weight = self.token_embedding.weight
    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    # ========== Create seq2seq modules ==========

    def _get_encoder(self) -> SeqEncoder:
        return SeqEncoder(
            self._config.encoder,
            self.token_embedding,
            self._vocabulary,
            len(self._vocabulary.label_to_id),
            self._vocabulary.label_to_id[PAD],
        )

    def _get_decoder(self) -> SeqDecoder:
        return SeqDecoder(
            self._config.decoder,
            self.token_embedding,
            self._vocabulary,
            self._config.encoder.embedding_size,
            len(self._vocabulary.label_to_id),
        )

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._config.hyper_parameters, self.parameters())

    def forward(  # type: ignore
        self,
        samples: PathBatch,
        batch_size: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        # [max path length; totul paths, encoder_size],[max path length; batch size]
        batched_features, attention_mask = self.encoder(
            samples.contexts,
            samples.contexts_per_label,
            samples.paths,
            samples.paths_per_label)
        # [max target length; batch size]
        target_embedding = self.encoder.token_embedding(target_sequence)
        output = self.decoder(batched_features, attention_mask, target_sequence.shape[0]-1, target_embedding)
        return output

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate cross entropy with ignoring PAD index

        :param logits: [seq length; batch size; vocab size]
        :param labels: [seq length+1; batch size]
        :return: [1]
        """
        batch_size = labels.shape[-1]
        # [batch size; vocab size; seq length]
        _logits = logits.permute(1, 2, 0)
        # [batch size; seq length]
        _labels = labels.permute(1, 0)
        # [batch size; seq length]
        loss = F.cross_entropy(_logits, _labels, reduction="none")
        # [batch size; seq length]
        mask = _labels != self._vocabulary.label_to_id[PAD]
        # [batch size; seq length]
        loss = loss * mask
        # [1]
        loss = loss.sum() / batch_size
        return loss

    # ========== Model step ==========

    def training_step(self, batch: PathBatch, batch_idx: int) -> Dict:  # type: ignore
        # [seq length; batch size; vocab size]
        logits = self(batch, batch.labels.shape[0], batch.labels)
        # [seq length; batch size]
        batch.labels = batch.labels[1:]

        loss = self._calculate_loss(logits, batch.labels)
        prediction = logits.argmax(-1)

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        batch_metric = statistic.update_statistic(batch.labels, prediction)

        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss.item()}
        for key, value in batch_metric.items():
            log[f"train/{key}"] = value
        self.log_dict(log)
        self.log("f1", batch_metric["f1"], prog_bar=True, logger=False)

        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: PathBatch, batch_idx: int) -> Dict:  # type: ignore
        self.decoder.teacher_forcing = 0
        self.eval()
        # [seq length; batch size; vocab size]
        logits = self(batch, batch.labels.shape[0], batch.labels)
        # [seq length; batch size]
        labels = batch.labels[1:]
        loss = self._calculate_loss(logits, labels)
        prediction = logits.argmax(-1)

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        statistic.update_statistic(labels, prediction)
        self.decoder.teacher_forcing = self._config.decoder.teacher_forcing
        self.train()
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: PathBatch, batch_idx: int) -> Dict:  # type: ignore
        return self.validation_step(batch, batch_idx)

    # ========== On epoch end ==========

    def _shared_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            statistic = PredictionStatistic.create_from_list([out["statistic"] for out in outputs])
            epoch_metrics = statistic.get_metric()
            log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": mean_loss}
            for key, value in epoch_metrics.items():
                log[f"{group}/{key}"] = value
            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)
            self.log(f"{group}_f1", epoch_metrics["f1"])

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test")
