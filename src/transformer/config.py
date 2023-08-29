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
    pretrained_tokenizer: str
    max_token_length: int
    label_smoothing: float
    sched_warmup: int
    grad_accum: int


transformer_cfg = TransformerConfig(
    n_encoders=6,
    n_decoders=6,
    vocab_sz=30000,
    emb_sz=512,
    ff_sz=512 * 4,
    n_heads=8,
    head_sz=512 // 8,
    pdrop=0.1,
)

train_cfg = TrainConfig(
    seed=None,
    train_bs=64,
    val_bs=64,
    test_bs=64,
    num_workers=8,
    n_epochs=15,
    pretrained_tokenizer="pretrained_tokenizers/bart_bpe_opus_en_id_30000",
    max_token_length=64,  # most sentences are under 64 tokens
    label_smoothing=0.1,
    sched_warmup=4000,
    grad_accum=2,
)
