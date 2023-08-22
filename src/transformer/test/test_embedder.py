import math

import pytest
import torch
from torch import nn

from model import Embedder


@pytest.mark.parametrize(
    "vocab_sz, emb_sz, pdrop, bs, seq",
    [
        (30000, 384, 0.2, 6, 512),
        (10000, 100, 0.5, 12, 30),
        (12345, 45, 0.3, 6, 10),
        (1, 1, 0.7, 1, 1),
    ],
)
def test_embedder_shapes(vocab_sz, emb_sz, pdrop, bs, seq):
    embedder = Embedder(vocab_sz=vocab_sz, emb_sz=emb_sz, pdrop=pdrop)
    ctx_input_ids = torch.randint(0, vocab_sz, (bs, seq))
    ctx = embedder(ctx_input_ids)
    assert ctx.size() == (bs, seq, emb_sz)


def test_embedder_dropout_applied():
    vocab_sz = 30000
    emb_sz = 384
    pdrop = 1.0
    embedder = Embedder(vocab_sz=vocab_sz, emb_sz=emb_sz, pdrop=pdrop).train()
    bs = 6
    seq = 37
    ctx_input_ids = torch.randint(0, vocab_sz, (bs, seq))
    ctx = embedder(ctx_input_ids)
    assert torch.all(ctx == 0.0)


def test_embedder_pos_encoding_applied():
    vocab_sz = 30000
    emb_sz = 384
    pdrop = 0.0
    embedder = Embedder(vocab_sz=vocab_sz, emb_sz=emb_sz, pdrop=pdrop)
    # force embedding to be 0
    embedder.token_emb_table.weight = nn.Parameter(
        torch.zeros_like(embedder.token_emb_table.weight)
    )
    bs = 6
    seq = 37
    ctx_input_ids = torch.randint(0, vocab_sz, (bs, seq))
    ctx = embedder(ctx_input_ids)
    assert torch.any(ctx != 0.0)


def _pos_encoding_torch(max_len: int, d_model: int):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # NOTE: this cannot handle odd max_len
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    print(position.size(), div_term.size(), pe.size())
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe


@pytest.mark.parametrize(
    "emb_sz, seq",
    [
        (384, 512),
        (100, 30),
        (44, 10),
        (2, 2),
    ],
)
def test_pos_encoding_vs_torch(emb_sz, seq):
    bs = 1  # squeezable outputs
    pdrop = 0.0  # guarantee no dropout applied
    vocab_sz = 30000
    embedder = Embedder(vocab_sz=vocab_sz, emb_sz=emb_sz, pdrop=pdrop)
    # force embedding to be 0
    embedder.token_emb_table.weight = nn.Parameter(
        torch.zeros_like(embedder.token_emb_table.weight)
    )
    ctx_input_ids = torch.randint(0, vocab_sz, (bs, seq))
    ctx = embedder(ctx_input_ids).squeeze()
    ctx_from_torch = _pos_encoding_torch(max_len=seq, d_model=emb_sz).squeeze()
    print((ctx - ctx_from_torch).abs().max())
    assert torch.allclose(ctx, ctx_from_torch, atol=1e-4)
