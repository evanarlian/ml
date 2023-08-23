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
