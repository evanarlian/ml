from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler


class YoloScheduler(LRScheduler):
    def __init__(
        self, optimizer, epochs, steps_per_epoch, last_epoch=-1, verbose=False
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.curr_step = -1  # -1 because below init will actually call step once
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        self.curr_step += 1
        curr_epoch = int(self.curr_step / self.steps_per_epoch)
        if curr_epoch < 1:
            # this is lerp
            first_epoch_ratio = self.curr_step / self.steps_per_epoch
            lr = 1e-3 + (1e-2 - 1e-3) * first_epoch_ratio
        elif curr_epoch < 75:
            lr = 1e-2
        elif curr_epoch < 105:
            lr = 1e-3
        elif curr_epoch < 135:
            lr = 1e-4
        else:
            # actually yolov1 does not use more than 135 epochs
            lr = 1e-4
        return [lr] * len(self.base_lrs)


def main():
    epochs = 135
    steps_per_epoch = 10
    model = nn.Linear(1, 1)
    # lr set on optimizer does not matter because of scheduler's init
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    scheduler = YoloScheduler(optimizer, epochs, steps_per_epoch)
    # see just a bit
    for epoch in range(5):
        for step in range(steps_per_epoch):
            print(f"{epoch=} {step=} {scheduler.get_last_lr()}")
            optimizer.step()
            scheduler.step()


if __name__ == "__main__":
    main()
