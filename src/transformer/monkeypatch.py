from functools import partial
from types import MethodType

import torch
from torch import Tensor

from model import MultiHeadAttention, Transformer, TransformerConfig


def _patched_sdpa(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, attn_dict: dict):
    attn = q @ k.transpose(-1, -2) / (self.head_sz**0.5)
    attn = attn.masked_fill(~mask, -5e4)
    attn = self.attn_drop(attn.softmax(-1))
    attn_dict[hash(self)] = attn.detach().cpu()  # monkey patched!
    out = attn @ v
    return out


def patch_and_get_attn(model: Transformer, model_inputs: dict[str, Tensor]):
    # patch model so it we can get attention
    hash2attn: dict[int, Tensor] = {}
    hash2name: dict[int, str] = {}
    patched_sdpa = partial(_patched_sdpa, attn_dict=hash2attn)
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            module.sdpa = MethodType(patched_sdpa, module)
            hash2name[hash(module)] = name
    # do the forward
    model(**model_inputs)
    # remap the dicts so the return value is cleaner
    name2attn: dict[str, Tensor] = {}
    for hsh, name in hash2name.items():
        name2attn[name] = hash2attn[hsh]
    return name2attn


def main():
    # make model
    n_encoders = 5
    n_decoders = 4
    vocab_sz = 30000
    emb_sz = 384
    ff_sz = 4 * emb_sz
    n_heads = 8
    head_sz = emb_sz // n_heads
    pdrop = 0.2
    cfg = TransformerConfig(
        n_encoders, n_decoders, vocab_sz, emb_sz, ff_sz, n_heads, head_sz, pdrop
    )
    model = Transformer(cfg)

    # fake inputs
    bs = 6
    seq_ctx, seq_tgt = 47, 35  # imagine a machine translation task
    ctx_input_ids = torch.randint(0, vocab_sz, (bs, seq_ctx))
    ctx_pad_mask = torch.randint(0, 2, (bs, seq_ctx))
    tgt_input_ids = torch.randint(0, vocab_sz, (bs, seq_tgt))
    tgt_pad_mask = torch.randint(0, 2, (bs, seq_tgt))
    model_inputs = {
        "ctx_input_ids": ctx_input_ids,
        "ctx_pad_mask": ctx_pad_mask,
        "tgt_input_ids": tgt_input_ids,
        "tgt_pad_mask": tgt_pad_mask,
    }

    # monkey patch and get attn
    model.eval()
    attn_dict = patch_and_get_attn(model, model_inputs)
    attn_dict2 = patch_and_get_attn(model, model_inputs)
    for k, v in attn_dict.items():
        print(v.size(), k)

    # this function is idempotent, same input, same attentions
    assert all(
        torch.all(v1 == v2) for v1, v2 in zip(attn_dict.values(), attn_dict2.values())
    ), "patching is not idempotent"
    # see that the number of entries in attn_dict is (n_encoders + 2*n_decoders)
    assert len(attn_dict) == (n_encoders + 2 * n_decoders)


if __name__ == "__main__":
    main()
