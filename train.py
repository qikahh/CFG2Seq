import os

from omegaconf import OmegaConf
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from os.path import join
from typing import Tuple

import hydra
import sys
import argparse
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from datamodule import PathDataModule
from model import CFGNode2Seq, Seq2Seq, Node2Seq, CFGSeq2Seq, RNN2Seq, CFGRNN2Seq
from utils.callback import UploadCheckpointCallback, PrintEpochResultCallback
from utils.common import print_config, filter_warnings
from utils.vocabulary import Vocabulary


def get_cfgnode2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = CFGNode2Seq(config, vocabulary)
    data_module = PathDataModule(config, vocabulary)
    return model, data_module

def get_seq2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = Seq2Seq(config, vocabulary)
    data_module = PathDataModule(config, vocabulary)
    return model, data_module

def get_node2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = Node2Seq(config, vocabulary)
    data_module = PathDataModule(config, vocabulary)
    return model, data_module

def get_cfgseq2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = CFGSeq2Seq(config, vocabulary)
    data_module = PathDataModule(config, vocabulary)
    return model, data_module

def get_rnn2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = RNN2Seq(config, vocabulary)
    data_module = PathDataModule(config, vocabulary)
    return model, data_module

def get_cfgrnn2seq(config: DictConfig, vocabulary: Vocabulary) -> Tuple[LightningModule, LightningDataModule]:
    model = CFGRNN2Seq(config, vocabulary)
    data_module = PathDataModule(config, vocabulary)
    return model, data_module

def train(config_name: str, checkpoint=None, with_train=True):
    config = OmegaConf.load(join('configs',config_name+'.yaml'))

    filter_warnings()
    print_config(config)
    seed_everything(config.seed)

    known_models = {"cfgnode2seq": get_cfgnode2seq, "seq2seq":get_seq2seq, "node2seq":get_node2seq, "cfgseq2seq": get_cfgseq2seq, 
                        "rnn2seq": get_rnn2seq, "cfgrnn2seq": get_cfgrnn2seq}
    if config.name not in known_models:
        print(f"Unknown model: {config.name}, try on of {known_models.keys()}")
    
    vocabulary = Vocabulary.load_vocabulary(join(config.data_folder, config.dataset.name, config.vocabulary_name))
    model, data_module = known_models[config.name](config, vocabulary)
    if checkpoint != None:
        model = model.load_from_checkpoint(checkpoint)

    # define logger
    wandb_logger = WandbLogger(
        name=f"{config.name}", save_dir=join("Wandb",f"{config.name}"), project=f"{config.project_name}-{config.dataset.name}", log_model=True, offline=config.log_offline
    )
    wandb_logger.watch(model)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{val_f1:.4f}",
        monitor='val_f1',
        period=config.save_every_epoch,
        save_top_k=3,
        mode='max'
    )
    upload_checkpoint_callback = UploadCheckpointCallback(wandb_logger.experiment.dir)
    # define early stopping callback
    #early_stopping_callback = EarlyStopping(
    #    patience=config.hyper_parameters.patience, monitor="val_f1", verbose=True, mode="max"
    #)
    # define callback for printing intermediate result
    print_epoch_result_callback = PrintEpochResultCallback("train", "val")
    # use gpu if it exists

    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.val_every_epoch,
        log_every_n_steps=config.log_every_epoch,
        logger=wandb_logger,
        gpus=gpu,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        callbacks=[
            lr_logger,
            # early_stopping_callback,
            checkpoint_callback,
            # upload_checkpoint_callback,
            print_epoch_result_callback,
        ],
        resume_from_checkpoint=checkpoint,
    )
    if with_train:
        trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str, default='cfgrnn2seq-DeGraphCS', help="config file")
    parser.add_argument("-c", "--checkpoint", type=str, default='Wandb/seq2seq/wandb/run-20210508_105946-tatwcl1t/files/epoch=09-val_f1=0.3707.ckpt', help="checkpoint file")
    args = parser.parse_args()
    train(args.yaml)#, args.checkpoint, False)
