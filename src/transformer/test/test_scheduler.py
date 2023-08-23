import pytest
from torch import nn, optim

from scheduler import TransformerScheduler


@pytest.mark.parametrize(
    "emb_sz, warmup_steps, n_iters",
    [
        (384, 4000, 2000),
        (200, 100, 99),
        (100, 5, 3),
    ],
)
def test_scheduler_warmup_applied(emb_sz, warmup_steps, n_iters):
    # test when n_iters < warmup, the lr is always increasing
    model = nn.Linear(10, 3)
    optimizer = optim.Adam(model.parameters())
    sched = TransformerScheduler(optimizer, emb_sz=emb_sz, warmup_steps=warmup_steps)
    lrs = []
    for i in range(n_iters):
        optimizer.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert all(a < b for a, b in zip(lrs[:-1], lrs[1:]))


@pytest.mark.parametrize(
    "emb_sz, warmup_steps, n_iters",
    [
        (384, 4000, 15000),
        (200, 100, 300),
        (100, 5, 10),
    ],
)
def test_scheduler_decay_applied(emb_sz, warmup_steps, n_iters):
    # check lrs after warmup must always decreasing
    model = nn.Linear(10, 3)
    optimizer = optim.Adam(model.parameters())
    sched = TransformerScheduler(optimizer, emb_sz=emb_sz, warmup_steps=warmup_steps)
    lrs = []
    for i in range(n_iters):
        optimizer.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    truncated = lrs[warmup_steps + 2 :]
    assert all(a > b for a, b in zip(truncated[:-1], truncated[1:]))


def test_optimizer_lr_equal_scheduler_lr():
    model = nn.Linear(10, 3)
    optimizer = optim.Adam(model.parameters())
    sched = TransformerScheduler(optimizer, emb_sz=384, warmup_steps=1000)
    lr_from_opt = optimizer.param_groups[0]["lr"]
    lr_from_sched = sched.get_last_lr()[0]
    assert lr_from_opt == lr_from_sched


def test_optimizer_first_time_is_not_default_lr():
    default_lr = 7.0
    model = nn.Linear(10, 3)
    optimizer = optim.Adam(model.parameters(), lr=default_lr)
    sched = TransformerScheduler(optimizer, emb_sz=384, warmup_steps=1000)
    lr_from_opt = optimizer.param_groups[0]["lr"]
    assert lr_from_opt != default_lr
