from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch import nn, optim
from torchmetrics.text import Perplexity, SacreBLEUScore
from transformers import BartTokenizer

from config import TrainConfig, TransformerConfig, train_cfg, transformer_cfg
from dataset import OpusEnId
from model import Transformer
from scheduler import TransformerScheduler
from utils import get_version, translate


class LitTransformer(L.LightningModule):
    def __init__(
        self,
        transformer_cfg: TransformerConfig,
        train_cfg: TrainConfig,
        tokenizer_dir: Path,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transformer_cfg = transformer_cfg
        self.train_cfg = train_cfg
        self.model = Transformer(transformer_cfg)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_dir)
        # metrics
        self.train_ppl = Perplexity()
        self.val_ppl = Perplexity()
        self.val_bleu = SacreBLEUScore()
        self.test_ppl = Perplexity()
        self.test_bleu = SacreBLEUScore()

    def _one_step(self, batch):
        logits = self.model(
            batch["ctx_input_ids"],
            batch["ctx_pad_mask"],
            batch["tgt_input_ids"],
            batch["tgt_pad_mask"],
        )
        loss = F.cross_entropy(logits.transpose(-1, -2), batch["labels"])
        bs = len(batch["ctx_input_ids"])
        # because batch dict contains str, pass batch size in self.log
        return (logits, loss, bs)

    def training_step(self, batch, batch_idx):
        logits, loss, bs = self._one_step(batch)
        self.train_ppl(logits.float(), batch["labels"])
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/perplexity", self.train_ppl, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, bs = self._one_step(batch)
        translated = translate(
            self.model,
            self.tokenizer,
            max_gen_length=self.train_cfg.max_token_length,
            text=batch["ctx_text"],
        )
        self.val_bleu(translated, [batch["tgt_text"]])
        self.val_ppl(logits.float(), batch["labels"])
        self.log("val/loss", loss, prog_bar=True, batch_size=bs)
        self.log("val/perplexity", self.val_ppl, prog_bar=True, batch_size=bs)
        self.log("val/sacrebleu", self.val_bleu, prog_bar=True, batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx):
        logits, loss, bs = self._one_step(batch)
        translated = translate(
            self.model,
            self.tokenizer,
            max_gen_length=self.train_cfg.max_token_length,
            text=batch["ctx_text"],
        )
        self.test_bleu(translated, [batch["tgt_text"]])
        self.test_ppl(logits.float(), batch["labels"])
        self.log("test/loss", loss, batch_size=bs)
        self.log("test/perplexity", self.test_ppl, batch_size=bs)
        self.log("test/sacrebleu", self.test_bleu, batch_size=bs)
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


def main():
    # TODO remove v_num in progress bar? https://lightning.ai/docs/pytorch/latest/extensions/logging.html#progress-bar
    # TODO wandb state is still finished when killed (ctrl-c)
    # TODO model checkpoint callback, best loss, best bleu
    # TODO make new wandb
    # setup
    train_cfg.seed = L.seed_everything(train_cfg.seed)  # seed will be selected if None
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
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    loggers = [
        CSVLogger(root_dir, version=exp_version),
        TensorBoardLogger(root_dir, version=exp_version),
        WandbLogger(project="shitty_transformer", save_dir=root_dir),
    ]
    trainer = L.Trainer(
        accelerator="gpu",
        # accelerator="cpu",
        precision="16-mixed",
        # fast_dev_run=10,
        max_epochs=train_cfg.n_epochs,
        val_check_interval=2000,
        default_root_dir=root_dir,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(lit_model)
    trainer.test(lit_model)


if __name__ == "__main__":
    main()
