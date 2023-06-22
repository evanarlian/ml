from pathlib import Path

import torch
import torchvision.transforms as T
from alexnet import AlexNet as MyAlexNet
from dataset import ImageNetMini
from torch import Tensor, nn, optim
from tqdm.auto import tqdm


def calc_topk_acc(y_true: Tensor, logits: Tensor, k: int) -> float:
    _, indices = logits.topk(k=k)
    return (indices == y_true.unsqueeze(1)).sum(-1).double().mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path("src/alexnet/alexnet.pt")  # for continue training

    # hparams from paper
    BATCH_SIZE = 128
    WEIGHT_DECAY = 0.0005
    LR = 0.01
    MOMENTUM = 0.9
    N_EPOCHS = 90

    # data TODO move to file
    train_aug = T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # TODO match paper
        ]
    )
    val_aug = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # TODO
        ]
    )
    train_ds = ImageNetMini("data/imagenet-mini/train/", train_aug)
    val_ds = ImageNetMini("data/imagenet-mini/val/", val_aug)

    # see models
    my_alexnet = MyAlexNet()
    if save_path.exists():
        print(f"Continue training from {save_path}")
        my_alexnet.load_state_dict(torch.load(save_path))
    my_alexnet.debug_forward(torch.randn(5, 3, 224, 224))

    # train model
    my_alexnet.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(my_alexnet.parameters(), LR, MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1)
    train_loader = train_ds.create_dataloader(BATCH_SIZE, 4, shuffle=True)
    val_loader = val_ds.create_dataloader(BATCH_SIZE, 4, shuffle=False)
    train_loss_counter = []
    val_loss_counter = []

    for epoch in range(1, N_EPOCHS + 1):
        # train
        my_alexnet.train()
        curr_losses = []
        for x, label in tqdm(train_loader, desc="train", leave=False):
            x = x.to(device)
            label = label.to(device)
            logits = my_alexnet(x)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            opt.zero_grad()
            curr_losses.append(loss.item())
        train_loss_counter.append(torch.tensor(curr_losses).mean().item())

        # val
        my_alexnet.eval()
        curr_losses = []
        curr_top1_acc = []
        curr_top5_acc = []
        for x, label in tqdm(val_loader, desc="val", leave=False):
            x = x.to(device)
            label = label.to(device)
            with torch.no_grad():
                logits = my_alexnet(x)
            loss = loss_fn(logits, label)
            curr_losses.append(loss.item())
            curr_top1_acc.append(calc_topk_acc(label, logits, k=1))
            curr_top5_acc.append(calc_topk_acc(label, logits, k=5))
        curr_losses = torch.tensor(curr_losses).mean().item()
        val_loss_counter.append(curr_losses)
        curr_top1_acc = torch.tensor(curr_top1_acc).mean().item()
        curr_top5_acc = torch.tensor(curr_top5_acc).mean().item()
        sched.step(curr_losses)

        # logging
        print(
            f"{epoch}/{N_EPOCHS} train_loss={train_loss_counter[-1]:.5f} val_loss={val_loss_counter[-1]:.5f} top1={curr_top1_acc:.5f} top5={curr_top5_acc:.5f}"
        )

    # save
    torch.save(my_alexnet.state_dict(), save_path)
    print(f"Training finished and weight saved to {save_path}")


if __name__ == "__main__":
    main()
