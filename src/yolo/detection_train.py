import warnings
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch import optim
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import detection_config as cfg
from detection_trainer import DetectorTrainer
from pascalvoc import build_pascalvoc
from yolo_loss import YoloLoss
from yolo_model import YoloDetection
from yolo_scheduler import YoloScheduler

# filter out the broadcasting warning in mse loss
warnings.filterwarnings(
    "ignore", message="Using a target size .* that is different to the input size .*"
)


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
            total_limit=1,  # this is for storing the latest state
        ),
        step_scheduler_with_optimizer=cfg.STEP_SCHED_WITH_OPT,
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
    Path(accelerator.project_dir).mkdir(parents=True)
    return accelerator


def freeze_backbone_groups(model: YoloDetection, until: int):
    # group 1-6, so i starts at 1
    for i, (name, child) in enumerate(model.backbone.named_children(), 1):
        if i <= until:
            child.requires_grad_(False)
            print(f"🧊 Freezing {name}")
        else:
            print(f"_ Skip freezing {name}")


def main():
    if cfg.SEED is not None:
        set_seed(cfg.SEED)
    project_dir = Path("src/yolo/")
    accelerator = setup_accelerate(project_dir, "yolo_detection", cfg.hparams)

    # load data
    train_dataset, val_dataset, class2id, id2class = build_pascalvoc(
        "data/VOC2012",
        cfg.INCLUDE_DIFFICULT,
        cfg.S,
        cfg.B,
        cfg.C,
    )
    train_loader = train_dataset.create_dataloader(cfg.TRAIN_BS, True, cfg.N_WORKERS)
    val_loader = val_dataset.create_dataloader(cfg.VAL_BS, False, cfg.N_WORKERS)

    # prepare training
    model = YoloDetection(B=cfg.B, C=cfg.C, use_sigmoid=cfg.USE_SIGMOID)
    if cfg.BACKBONE_PATH is not None:
        model.backbone.load_state_dict(torch.load(cfg.BACKBONE_PATH))
    freeze_backbone_groups(model, cfg.FREEZE_BACKBONE_UNTIL)
    criterion = YoloLoss(S=cfg.S, B=cfg.B, C=cfg.C)

    # optimizer
    if cfg.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.MAX_LR,
            weight_decay=cfg.WD,
            momentum=cfg.SGD_MOMENTUM,
            nesterov=cfg.SGD_NESTEROV,
        )
    elif cfg.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.MAX_LR, weight_decay=cfg.WD)
    else:
        raise ValueError("optim choice not supported")

    # scheduler
    if cfg.SCHEDULER == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.MAX_LR,
            epochs=cfg.N_EPOCHS,
            steps_per_epoch=len(train_loader),
            anneal_strategy="linear",
        )
    elif cfg.SCHEDULER == "yolo":
        if cfg.N_EPOCHS < 135:
            warnings.warn(
                f"🔔 Yolov1 uses 135 epochs. By using less than that ({cfg.N_EPOCHS}), "
                "the learning rate scheduler will not be fully completed."
            )
        scheduler = YoloScheduler(
            optimizer,
            epochs=cfg.N_EPOCHS,
            steps_per_epoch=len(train_loader),
            divider=cfg.YOLO_SCHED_DIVIDER,
        )
    elif cfg.SCHEDULER is None:
        # noop scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    else:
        raise ValueError("scheduler choice not supported")

    # metrics
    map_metric = MetricCollection([MeanAveragePrecision(box_format="cxcywh")])
    train_map_metric = map_metric.clone(prefix="train/")
    val_map_metric = map_metric.clone(prefix="val/")
    val_loss_metric = MeanMetric()

    # accelerator wraps everything
    (
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_map_metric,
        val_map_metric,
        val_loss_metric,
    ) = accelerator.prepare(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_map_metric,
        val_map_metric,
        val_loss_metric,
    )

    # train
    trainer = DetectorTrainer(
        accelerator,
        model,
        criterion,
        optimizer,
        scheduler,
        class2id,
        id2class,
        cfg.hparams,
        train_map_metric,
        val_map_metric,
        val_loss_metric,
    )
    trainer.fit(train_loader, val_loader, cfg.N_EPOCHS)
    # trainer.overfit_one_batch(train_loader, scheduler_step=True)

    accelerator.end_training()  # for trackers finalization


if __name__ == "__main__":
    main()
