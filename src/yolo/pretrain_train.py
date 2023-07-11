from datetime import datetime
from pathlib import Path

import pretrain_config as cfg
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from imagenet100 import build_imagenet100
from pretrain_trainer import BackboneTrainer
from torch import nn, optim
from torchmetrics import (
    Accuracy,
    F1Score,
    MeanMetric,
    MetricCollection,
    Precision,
    Recall,
)
from yolo_model import YoloPretraining


def setup_accelerate(project_root: Path, project_name: str, hparams: dict):
    # config all accelerate stuffs
    accelerate_logs_dir = project_root / "accelerate_logs"
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with=["wandb"],  # can be anything that inherits GeneralTracker
        # dynamo_backend="inductor",
        project_config=ProjectConfiguration(
            project_dir=accelerate_logs_dir / f"exp_{now}",
            automatic_checkpoint_naming=True,
            total_limit=1,  # this is a hack for storing the best model
        ),
        step_scheduler_with_optimizer=True,  # TODO this is not always true
    )
    accelerator.init_trackers(
        project_name=project_name,  # your_name/<project_name> in wandb website
        config=hparams,  # hparams and/or one-time-per-run data
        init_kwargs={
            "wandb": {  # will be fed to wandb.init(), see https://docs.wandb.ai/ref/python/init
                # "project": project_name,  # no need, the same as above 'project_name'
                # "name": "my-run-07",  # will be assigned if not given
                # "notes": "my notes",  # markdown, can be edited in wandb website
                # "config": hparams,  # no need, the same as above 'config'
                "dir": accelerate_logs_dir,  # local wandb dir ('wandb' will be auto appended)
            }
        },
    )
    return accelerator


def main():
    if cfg.SEED is not None:
        set_seed(cfg.SEED)
    project_dir = Path("src/yolo/")
    accelerator = setup_accelerate(project_dir, "yolo_pretraining", cfg.hparams)

    # load data
    train_dataset, val_dataset = build_imagenet100()
    train_loader = train_dataset.create_dataloader(cfg.TRAIN_BS, True, cfg.N_WORKERS)
    val_loader = val_dataset.create_dataloader(cfg.VAL_BS, False, cfg.N_WORKERS)

    # prepare training
    total_steps = len(train_loader) * cfg.N_EPOCHS
    model = YoloPretraining(n_classes=cfg.N_CLASSES)
    criterion = nn.CrossEntropyLoss()

    if cfg.OPTIMIZER == "sgd":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.MAX_LR)
    elif cfg.OPTIMIZER == "adamw":
        optimizer = optim.SGD(model.parameters(), lr=cfg.MAX_LR, momentum=0.9)
    else:
        raise ValueError("optim choice not supported")

    if cfg.SCHEDULER == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.MAX_LR, total_steps=total_steps
        )
    elif cfg.SCHEDULER is None:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )  # noop scheduler
    else:
        raise ValueError("scheduler choice not supported")

    # metrics, have to use dict style if containing >2 of the same metrics
    metrics = MetricCollection(
        {
            "accuracy_top1": Accuracy("multiclass", num_classes=cfg.N_CLASSES, top_k=1),
            "accuracy_top5": Accuracy("multiclass", num_classes=cfg.N_CLASSES, top_k=5),
            "precision": Precision("multiclass", num_classes=cfg.N_CLASSES),
            "recall": Recall("multiclass", num_classes=cfg.N_CLASSES),
            "f1_score": F1Score("multiclass", num_classes=cfg.N_CLASSES),
        }
    )
    train_metrics = metrics.clone(prefix="train/")
    val_metrics = metrics.clone(prefix="val/")
    val_loss_metric = MeanMetric()

    # accelerator wraps everything
    (
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_metrics,
        val_metrics,
        val_loss_metric,
    ) = accelerator.prepare(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_metrics,
        val_metrics,
        val_loss_metric,
    )

    # train
    trainer = BackboneTrainer(
        accelerator,
        model,
        criterion,
        optimizer,
        scheduler,
        train_metrics,
        val_metrics,
        val_loss_metric,
    )
    trainer.fit(train_loader, val_loader, cfg.N_EPOCHS)
    # trainer.overfit_one_batch(train_loader)
    # 100 1.0833674669265747 maxlr=0.1
    # 100 1.9721401258721016e-05 maxlr=0.01
    # 100 1.2867069017374888e-05 maxlr=0.03
    # 100 3.648481651907787e-05 maxlr=0.001

    accelerator.end_training()  # for trackers finalization


if __name__ == "__main__":
    main()
