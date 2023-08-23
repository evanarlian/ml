from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler


class TransformerScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        emb_sz: int,
        warmup_steps: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.emb_sz = emb_sz
        self.warmup_steps = warmup_steps
        # init will actually step (get_lr) once, so do it last
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self._step_count
        lrate = (self.emb_sz**-0.5) * min(
            step**-0.5, step * self.warmup_steps**-1.5
        )
        return [lrate] * len(self.base_lrs)


def main():
    model = nn.Linear(10, 3)
    optimizer = optim.Adam(model.parameters())
    sched = TransformerScheduler(optimizer, emb_sz=384, warmup_steps=4000)
    lrs = []
    for i in range(30000):
        optimizer.step()
        sched.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        if i == 0:
            print("first lr", optimizer.param_groups[0]["lr"])
    import matplotlib.pyplot as plt

    plt.plot(lrs)
    plt.show()


if __name__ == "__main__":
    main()
