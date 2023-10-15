import torch
from torch import Tensor, nn

from config import TransformerConfig


class Embedder(nn.Module):
    def __init__(self, vocab_sz: int, emb_sz: int, pdrop: float):
        super().__init__()
        self.emb_sz = emb_sz
        self.token_emb_table = nn.Embedding(vocab_sz, emb_sz)
        self.embd_drop = nn.Dropout(pdrop)

    def positional(self, seq: int, n: int = 10000):
        positions = torch.arange(seq).unsqueeze(-1)  # 0 1 2 3 ... (col vector)
        pairs = torch.arange(self.emb_sz) // 2  # 0 0 1 1 2 2 ... (row vector)
        is_even = torch.arange(self.emb_sz) % 2 == 0  # 0 1 0 1 0 ... (row vector)
        sins = torch.sin(positions / (n ** (2 * pairs / self.emb_sz)))
        coses = torch.cos(positions / (n ** (2 * pairs / self.emb_sz)))
        return (is_even * sins) + (~is_even * coses)

    def forward(self, input_ids: Tensor) -> Tensor:
        bs, seq = input_ids.size()
        x = self.token_emb_table(input_ids) + self.positional(seq).to(input_ids.device)
        x = self.embd_drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_sz: int, n_heads: int, head_sz: int, pdrop: float):
        """
        Implements a general multihead attention. All this class care are:
        performing attention with given mask:
        * If no mask is required, pass an all True tensor
        * If multiple masks (i.e. causal + attn), pass the &'ed mask
        Also provides with useful methods as utils.

        Args:
            emb_sz (int): Embedding size
            n_heads (int): Number of attention heads
            head_sz (int): Size of each attention head
            pdrop (int): Dropout probability
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_sz = head_sz
        self.query = nn.Linear(emb_sz, n_heads * head_sz, bias=False)
        self.key = nn.Linear(emb_sz, n_heads * head_sz, bias=False)
        self.value = nn.Linear(emb_sz, n_heads * head_sz, bias=False)
        self.attn_drop = nn.Dropout(pdrop)
        self.proj = nn.Linear(n_heads * head_sz, emb_sz, bias=False)

    def sdpa(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        """
        Scaled dot product attention.

        Args:
            q (Tensor): query tensor. Size (..., seq_q, head_sz)
            k (Tensor): key tensor. Size (..., seq_k, head_sz)
            v (Tensor): value tensor. Size (..., seq_v, head_sz)
            mask (Tensor): Bool tensor that allows attention.

        Returns:
            Tensor: Attn multiplied with value. Size (..., seq_q, head_sz)
        """
        # attn shape will be (bs, n_heads, seq_q, seq_k)
        attn = q @ k.transpose(-1, -2) / (self.head_sz**0.5)
        attn = attn.masked_fill(~mask, -5e4)  # NOTE: if too small will fail fp16
        attn = self.attn_drop(attn.softmax(-1))
        out = attn @ v
        return out

    def make_causal_mask(self, seq: int, device: torch.device) -> Tensor:
        """Create causal mask that allows self-attention on current and past tokens"""
        return torch.ones(seq, seq, dtype=torch.bool, device=device).tril()

    def forward(self, xq: Tensor, xk: Tensor, xv: Tensor, mask: Tensor) -> Tensor:
        """
        Perform a multihead attention with projection

        Args:
            xq (Tensor): Tensor for query weight's input. Size (..., seq_q, emb_sz)
            xk (Tensor): Tensor for key weight's input. Size (..., seq_k, emb_sz)
            xv (Tensor): Tensor for value weight's input. Size (..., seq_v, emb_sz)
            mask (Tensor): Bool tensor that allows attention. The mask has to be
                broadcastable with q @ k.T, and you can combine them before passing
                to this method using & tensor operator, for example if you want to use
                cuasal mask and padding mask at the same time, mask = causal & attn.

        Returns:
            Tensor
        """
        # rules that must be right in every case (vanilla, causal, cross attn, etc)
        assert xk.size(-2) == xv.size(-2)
        bs, seq_q, seq_k, seq_v = xq.size(0), xq.size(-2), xk.size(-2), xv.size(-2)
        q = self.query(xq).view(bs, seq_q, self.n_heads, self.head_sz).transpose(-2, -3)
        k = self.key(xk).view(bs, seq_k, self.n_heads, self.head_sz).transpose(-2, -3)
        v = self.value(xv).view(bs, seq_v, self.n_heads, self.head_sz).transpose(-2, -3)
        out = self.sdpa(q, k, v, mask)
        # NOTE: below is using reshape because view cannot work on non-contiguous tensor
        out = out.transpose(-2, -3).reshape(bs, seq_q, self.n_heads * self.head_sz)
        out = self.proj(out)
        return out


# below are 3 different classes (flavor) of multihead attention
# they are separated so that most of the error will not be at runtime
# plus separation between general and specific implementation


class VanillaMHA(nn.Module):
    def __init__(self, emb_sz: int, n_heads: int, head_sz: int, pdrop: float):
        """Multihead attention with padding attention mask"""
        super().__init__()
        self.mha = MultiHeadAttention(emb_sz, n_heads, head_sz, pdrop)

    def forward(self, ctx: Tensor, ctx_pad_mask: Tensor) -> Tensor:
        # ctx (bs, seq_c, emb_sz)
        # ctx_pad_mask (bs, seq_c)
        # below code is to make a pairwise masking (bs, seq_c, seq_c)
        ctx_pad_mask = (ctx_pad_mask[:, :, None] * ctx_pad_mask[:, None, :]).bool()
        # compensate for n_heads by pre-unsqueezing
        ctx_pad_mask = ctx_pad_mask.unsqueeze(-3)  # (bs, 1, seq_c, seq_c)
        return self.mha(ctx, ctx, ctx, ctx_pad_mask)


class MaskedMHA(nn.Module):
    def __init__(self, emb_sz: int, n_heads: int, head_sz: int, pdrop: float):
        """Multihead attention with padding attention mask and causal mask"""
        super().__init__()
        self.mha = MultiHeadAttention(emb_sz, n_heads, head_sz, pdrop)

    def forward(self, tgt: Tensor, tgt_pad_mask: Tensor) -> Tensor:
        # tgt (bs, seq_t, emb_sz)
        # tgt_pad_mask (bs, seq_t)
        # below code is to make a pairwise masking (bs, seq_t, seq_t)
        tgt_pad_mask = (tgt_pad_mask[:, :, None] * tgt_pad_mask[:, None, :]).bool()
        causal_mask = self.mha.make_causal_mask(tgt.size(-2), tgt.device)
        combined_mask = tgt_pad_mask & causal_mask
        # compensate for n_heads by pre-unsqueezing
        combined_mask = combined_mask.unsqueeze(-3)  # (bs, 1, seq_c, seq_c)
        return self.mha(tgt, tgt, tgt, combined_mask)


class CrossMHA(nn.Module):
    def __init__(self, emb_sz: int, n_heads: int, head_sz: int, pdrop: float):
        """Multihead attention with cross attn, both encoder and decoder pad mask"""
        super().__init__()
        self.mha = MultiHeadAttention(emb_sz, n_heads, head_sz, pdrop)

    def forward(
        self, ctx: Tensor, ctx_pad_mask: Tensor, tgt: Tensor, tgt_pad_mask: Tensor
    ) -> Tensor:
        # ctx (bs, seq_c, emb_sz)
        # ctx_pad_mask (bs, seq_c)
        # tgt (bs, seq_t, emb_sz)
        # tgt_pad_mask (bs, seq_t)
        # below code is to make a pairwise masking (bs, seq_t, seq_c)
        tgt_ctx_pad_mask = (tgt_pad_mask[:, :, None] * ctx_pad_mask[:, None, :]).bool()
        # compensate for n_heads by pre-unsqueezing
        tgt_ctx_pad_mask = tgt_ctx_pad_mask.unsqueeze(-3)  # (bs, 1, seq_c, seq_c)
        return self.mha(tgt, ctx, ctx, tgt_ctx_pad_mask)


class FeedForward(nn.Module):
    def __init__(self, emb_sz: int, ff_sz: int):
        super().__init__()
        self.fc1 = nn.Linear(emb_sz, ff_sz)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_sz, emb_sz)

    def forward(self, x: Tensor) -> Tensor:
        # x is (bs, seq, emb_sz)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(
        self, emb_sz: int, n_heads: int, head_sz: int, ff_sz: int, pdrop: float
    ):
        super().__init__()
        # bottom part
        self.vanilla_mha = VanillaMHA(emb_sz, n_heads, head_sz, pdrop)
        self.resid_drop1 = nn.Dropout(pdrop)
        self.ln1 = nn.LayerNorm(emb_sz)
        # top part
        self.ffn = FeedForward(emb_sz, ff_sz)
        self.resid_drop2 = nn.Dropout(pdrop)
        self.ln2 = nn.LayerNorm(emb_sz)

    def forward(self, ctx: Tensor, ctx_pad_mask: Tensor) -> Tensor:
        # NOTE: many transformer implementations now use pre-layer norm
        # I use the original paper instead, because I want to replicate 1 to 1
        ctx = ctx + self.resid_drop1(self.vanilla_mha(ctx, ctx_pad_mask))
        ctx = self.ln1(ctx)
        ctx = ctx + self.resid_drop2(self.ffn(ctx))
        ctx = self.ln2(ctx)
        return ctx


class DecoderLayer(nn.Module):
    def __init__(
        self, emb_sz: int, n_heads: int, head_sz: int, ff_sz: int, pdrop: float
    ):
        super().__init__()
        # bottom part
        self.masked_mha = MaskedMHA(emb_sz, n_heads, head_sz, pdrop)
        self.resid_drop1 = nn.Dropout(pdrop)
        self.ln1 = nn.LayerNorm(emb_sz)
        # middle part
        self.cross_mha = CrossMHA(emb_sz, n_heads, head_sz, pdrop)
        self.resid_drop2 = nn.Dropout(pdrop)
        self.ln2 = nn.LayerNorm(emb_sz)
        # top part
        self.ffn = FeedForward(emb_sz, ff_sz)
        self.resid_drop3 = nn.Dropout(pdrop)
        self.ln3 = nn.LayerNorm(emb_sz)

    def forward(
        self, ctx: Tensor, ctx_pad_mask: Tensor, tgt: Tensor, tgt_pad_mask: Tensor
    ) -> Tensor:
        # fmt: off
        tgt = tgt + self.resid_drop1(self.masked_mha(tgt, tgt_pad_mask))
        tgt = self.ln1(tgt)
        tgt = tgt + self.resid_drop2(self.cross_mha(ctx, ctx_pad_mask, tgt, tgt_pad_mask))  # noqa: E501
        tgt = self.ln2(tgt)
        tgt = tgt + self.resid_drop3(self.ffn(tgt))
        tgt = self.ln3(tgt)
        # fmt: on
        return tgt


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # transformer encoder
        self.ctx_embedder = Embedder(
            vocab_sz=config.vocab_sz,
            emb_sz=config.emb_sz,
            pdrop=config.pdrop,
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayer(
                    emb_sz=config.emb_sz,
                    n_heads=config.n_heads,
                    head_sz=config.head_sz,
                    ff_sz=config.ff_sz,
                    pdrop=config.pdrop,
                )
                for _ in range(config.n_encoders)
            ]
        )
        # transformer decoder
        self.tgt_embedder = Embedder(
            vocab_sz=config.vocab_sz, emb_sz=config.emb_sz, pdrop=config.pdrop
        )
        self.decoders = nn.ModuleList(
            [
                DecoderLayer(
                    emb_sz=config.emb_sz,
                    n_heads=config.n_heads,
                    head_sz=config.head_sz,
                    ff_sz=config.ff_sz,
                    pdrop=config.pdrop,
                )
                for _ in range(config.n_decoders)
            ]
        )
        # final topmost layer
        self.final_fc = nn.Linear(config.emb_sz, config.vocab_sz)

    def forward(
        self,
        ctx_input_ids: Tensor,
        ctx_pad_mask: Tensor,
        tgt_input_ids: Tensor,
        tgt_pad_mask: Tensor,
    ) -> Tensor:
        ctx = self.ctx_embedder(ctx_input_ids)
        for e in self.encoders:
            ctx = e(ctx, ctx_pad_mask)
        tgt = self.tgt_embedder(tgt_input_ids)
        for d in self.decoders:
            tgt = d(ctx, ctx_pad_mask, tgt, tgt_pad_mask)
        tgt = self.final_fc(tgt)
        return tgt


def main():
    # simple sanity check
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
    logits = model(ctx_input_ids, ctx_pad_mask, tgt_input_ids, tgt_pad_mask)
    print(logits.size())
    print("sanity check complete!")


if __name__ == "__main__":
    main()
