from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_encoders: int  # number of encoder layers
    n_decoders: int  # number of decoder layers
    vocab_sz: int  # number of all possible token ids
    emb_sz: int  # embedding size for each token
    ff_sz: int  # embedding dimension of feedforward
    n_heads: int  # head size for each head in multihead attention
    head_sz: int  # number of attention heads
    pdrop: float  # dropout probability


@dataclass
class TrainConfig:
    seed: int | None
    train_bs: int
    val_bs: int
    test_bs: int
    num_workers: int
    n_epochs: int
    lr: float


transformer_cfg = TransformerConfig(
    n_encoders=4,
    n_decoders=4,
    vocab_sz=30000,
    emb_sz=256,
    ff_sz=256 * 4,
    n_heads=8,
    head_sz=256 // 8,
    pdrop=0.2,
)

train_cfg = TrainConfig(
    seed=None,
    train_bs=64,
    val_bs=64,
    test_bs=64,
    num_workers=4,
    n_epochs=10,
    lr=0.001,
)
