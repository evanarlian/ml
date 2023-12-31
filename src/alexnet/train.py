from pathlib import Path

import torch
from datasets import load_dataset
from torch import Tensor, nn, optim
from tqdm.auto import tqdm

from alexnet import AlexNet as MyAlexNet
from dataset import ImageNet, make_train_aug, make_val_aug


def calc_topk_acc(y_true: Tensor, logits: Tensor, k: int) -> float:
    _, indices = logits.topk(k=k)
    return (indices == y_true.unsqueeze(1)).sum(-1).double().mean().item()


def save_best(model: nn.Module, epoch: int, loss: float, save_folder: Path):
    for file in save_folder.glob("alexnet_best_*.pt"):
        file.unlink()
    # detect for compiled model, we want for the final model
    # to be able to be loaded from the checkpoint cleanly
    if hasattr(model, "_orig_mod"):
        state_dict = model._orig_mod.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_folder / f"alexnet_best_ep{epoch}_loss{loss:.3f}.pt")


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = Path("src/alexnet/ckpt/")
    save_folder.mkdir(parents=True, exist_ok=True)

    # hparams from paper
    BATCH_SIZE = 128
    WEIGHT_DECAY = 0.0005
    LR = 0.01
    MOMENTUM = 0.9
    N_EPOCHS = 90

    # get dataset
    ds = load_dataset("evanarlian/imagenet_1k_resized_256")
    train_aug = make_train_aug()
    val_aug = make_val_aug()  # use train's too to prevent data leak
    train_ds = ImageNet(ds["train"], train_aug)
    val_ds = ImageNet(ds["val"], val_aug)

    # load models
    my_alexnet = MyAlexNet()
    my_alexnet = torch.compile(my_alexnet)
    my_alexnet = my_alexnet.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(my_alexnet.parameters(), LR, MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=5, verbose=True
    )
    train_loader = train_ds.create_dataloader(BATCH_SIZE, 4, shuffle=True)
    val_loader = val_ds.create_dataloader(BATCH_SIZE, 4, shuffle=False)
    train_loss_counter = []
    val_loss_counter = []
    best_loss = float("inf")

    # train model
    for epoch in range(1, N_EPOCHS + 1):
        # train
        my_alexnet.train()
        curr_losses = []
        for x, label in (pbar := tqdm(train_loader, desc="train", leave=False)):
            x = x.to(device)
            label = label.to(device)
            logits = my_alexnet(x)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss = loss.item()
            curr_losses.append(loss)
            pbar.set_postfix({"loss": loss})
        train_loss_counter.append(torch.tensor(curr_losses).mean().item())

        # val
        my_alexnet.eval()
        curr_losses = []
        curr_top1_acc = []
        curr_top5_acc = []
        for x, label in (pbar := tqdm(val_loader, desc="val", leave=False)):
            x = x.to(device)
            label = label.to(device)
            with torch.no_grad():
                logits = my_alexnet(x)
            loss = loss_fn(logits, label).item()
            curr_losses.append(loss)
            pbar.set_postfix({"loss": loss})
            curr_top1_acc.append(calc_topk_acc(label, logits, k=1))
            curr_top5_acc.append(calc_topk_acc(label, logits, k=5))
        curr_losses = torch.tensor(curr_losses).mean().item()
        val_loss_counter.append(curr_losses)
        curr_top1_acc = torch.tensor(curr_top1_acc).mean().item()
        curr_top5_acc = torch.tensor(curr_top5_acc).mean().item()
        sched.step(curr_losses)

        # save
        if curr_losses < best_loss:
            print(f" > Found lower loss {curr_losses}")
            best_loss = curr_losses
            save_best(my_alexnet, epoch, curr_losses, save_folder)

        # logging
        print(
            f"{epoch}/{N_EPOCHS} train_loss={train_loss_counter[-1]:.5f} val_loss={val_loss_counter[-1]:.5f} top1={curr_top1_acc:.5f} top5={curr_top5_acc:.5f}"
        )


if __name__ == "__main__":
    main()
