from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch import nn, optim
from transformers import BartTokenizer

from config import TrainConfig, TransformerConfig, train_cfg, transformer_cfg
from dataset import OpusEnId
from model import Transformer
from scheduler import TransformerScheduler
from torchmetrics import Perplexity
from torchmetrics.text import BLEUScore

# TODO lr callbacks, wandb, log text and, so we need utils for decoding?


class LitTransformer(L.LightningModule):
    def __init__(
        self,
        transformer_cfg: TransformerConfig,
        train_cfg: TrainConfig,
        tokenizer_dir: Path,
    ):
        super().__init__()
        self.transformer_cfg = transformer_cfg
        self.train_cfg = train_cfg
        self.model = Transformer(transformer_cfg)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_dir)
        self.save_hyperparameters()

    def _one_step(self, batch):
        logits = self.model(
            batch["ctx_input_ids"],
            batch["ctx_pad_mask"],
            batch["tgt_input_ids"],
            batch["tgt_pad_mask"],
        )
        loss = F.cross_entropy(logits.transpose(-1, -2), batch["labels"])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._one_step(batch)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        scheduler = TransformerScheduler(
            optimizer,
            emb_sz=self.transformer_cfg.emb_sz,
            warmup_steps=self.train_cfg.sched_warmup,
        )
        # possible naming conventions if using single/multiple param group
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html
        # wandb trick is to enable "Group first prefix" to match tensorboard behavior
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "train/lr",
            },
        }

    def train_dataloader(self):
        train_ds = OpusEnId(
            "train", self.tokenizer, max_length=train_cfg.max_token_length
        )
        return train_ds.create_dataloader(
            batch_size=self.train_cfg.train_bs,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        val_ds = OpusEnId(
            "validation", self.tokenizer, max_length=train_cfg.max_token_length
        )
        return val_ds.create_dataloader(
            batch_size=self.train_cfg.train_bs,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        test_ds = OpusEnId(
            "test", self.tokenizer, max_length=train_cfg.max_token_length
        )
        return test_ds.create_dataloader(
            batch_size=self.train_cfg.train_bs,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            drop_last=False,
        )


def get_version(root_dir: Path, log_dir: str = "lightning_logs"):
    """Extract number from every version and find new max version"""
    max_ver = 0
    log_path = root_dir / log_dir
    if not log_path.exists():
        return 0
    for ver in (root_dir / log_dir).iterdir():
        try:
            max_ver = max(max_ver, int(ver.name.split("_")[-1]))
        except ValueError:
            pass
    return max_ver + 1


def main():
    # TODO remove v_num in progress bar?
    # TODO wandb state is still finished when killed (ctrl-c)
    # TODO model checkpoint callback
    # setup
    # print("🔥", transformer_cfg)
    # print("🔥", train_cfg)
    L.seed_everything(train_cfg.seed)
    root_dir = Path("src/transformer")
    tokenizer_dir = root_dir / train_cfg.pretrained_tokenizer
    torch.set_float32_matmul_precision("high")  # TODO training is slower with this
    exp_version = get_version(root_dir)

    # model
    lit_model = LitTransformer(
        transformer_cfg=transformer_cfg,
        train_cfg=train_cfg,
        tokenizer_dir=tokenizer_dir,
    )

    # trainer stuffs
    callbacks = [
        LearningRateMonitor("step"),
    ]

    loggers = [
        CSVLogger(root_dir, version=exp_version),
        TensorBoardLogger(root_dir, version=exp_version),
        WandbLogger(project="shitty_transformer", save_dir=root_dir),
    ]
    # TODO i kinda want to do dry run rn :((
    trainer = L.Trainer(
        accelerator="gpu",
        # accelerator="cpu",
        precision="16-mixed",
        # fast_dev_run=4,
        max_epochs=train_cfg.n_epochs,
        overfit_batches=0.0,  # default TODO does this enable weird shit?
        log_every_n_steps=50,  # default
        val_check_interval=2000,
        # check_val_every_n_epoch=None,
        default_root_dir=root_dir,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(lit_model)


if __name__ == "__main__":
    main()
