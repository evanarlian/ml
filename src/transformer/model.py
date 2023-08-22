from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# TODO pytest device
# TODO clean up shit in the main func
# TODO attention mask, this is for different sequence length, pytest material
# TODO kv_cache, later


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
    def __init__(
        self, flavor: str, emb_sz: int, n_heads: int, head_sz: int, pdrop: float
    ):
        """
        Implements multihead attention in 3 different flavors:
        * vanilla = used in the encoder part, qkv comes from previous layer
        * masked = used in the bottom part of the decoder, qkv comes from previous layer
        * cross = used in the upper part of the decoder, kv comes from encoder,
                  but only q comes from previous (decoder) layer

        Args:
            flavor (str): multihead attention type: {"vanilla", "masked", "cross"}
            emb_sz (int): Embedding size
            n_heads (int): Number of attention heads
            head_sz (int): Size of each attention head
            pdrop (int): Dropout probability
        """
        super().__init__()
        assert flavor in {"vanilla", "masked", "cross"}
        self.flavor = flavor
        self.n_heads = n_heads
        self.head_sz = head_sz
        self.query = nn.Linear(emb_sz, n_heads * head_sz, bias=False)
        self.key = nn.Linear(emb_sz, n_heads * head_sz, bias=False)
        self.value = nn.Linear(emb_sz, n_heads * head_sz, bias=False)
        self.attn_drop = nn.Dropout(pdrop)
        self.proj = nn.Linear(n_heads * head_sz, emb_sz)  # TODO bias here?

    def sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        causal_mask: Tensor | None = None,
    ):
        """
        Scaled dot product attention.

        Args:
            q (Tensor): query tensor.
            k (Tensor): key tensor.
            v (Tensor): value tensor.
            attn_mask (Tensor, optional): Indicator of non padding sequence element
            causal_mask (Tensor, optional): Mask that allows which token to communicate.

        Returns:
            Tensor: Attention multiplied with value tensor.
        """
        # q k v is all (bs, n_heads, seq, head_sz)
        bs, n_heads, seq, head_sz = q.size()
        attn = q @ k.transpose(-1, -2) / (head_sz**0.5)  # (bs, n_heads, seq, seq)
        if causal_mask is not None:
            attn = attn.masked_fill(~causal_mask, -torch.inf)
        if attn_mask is not None:
            # attn mask blocks whole rows, so if we use -inf, the output will be nan
            print("🔥", attn.size(), attn_mask.size())
            attn_mask = attn_mask.unsqueeze(-3)
            attn = attn.masked_fill(~attn_mask, -1e15)
        attn = self.attn_drop(attn.softmax(-1))
        out = (attn @ v).transpose(-2, -3).contiguous().view(bs, seq, n_heads * head_sz)
        return out

    def forward(
        self, x: Tensor, attn_mask: Tensor | None = None, context: Tensor | None = None
    ) -> Tensor:
        """
        Perform multihead attention according to the flavor.

        Args:
            x (Tensor): Tensor from previous layer
            attn_mask (Tensor, optional): Indicator of non padding sequence element
            context (Tensor, optional): Tensor from encoder, only used in flavor "cross"

        Returns:
            Tensor
        """
        bs, seq, emb_sz = x.size()
        q = self.query(x).view(bs, seq, self.n_heads, self.head_sz).transpose(-2, -3)
        # fmt: off
        if self.flavor == "cross":
            assert context is not None, "cross attention require context from encoder"
            ctx_seq = context.size(-2)  # NOTE: in cross, use context's seq for k and v
            k = self.key(context).view(bs, ctx_seq, self.n_heads, self.head_sz).transpose(-2, -3)  # noqa: E501
            v = self.value(context).view(bs, ctx_seq, self.n_heads, self.head_sz).transpose(-2, -3)  # noqa: E501
        else:
            assert context is None, "vanilla or masked attention does not require encoder context"  # noqa: E501
            k = self.key(x).view(bs, seq, self.n_heads, self.head_sz).transpose(-2, -3)
            v = self.value(x).view(bs, seq, self.n_heads, self.head_sz).transpose(-2, -3)  # noqa: E501
        # fmt: on
        causal_mask = None
        if self.flavor == "masked":
            # NOTE: we calculate mask on the fly because in original transformer paper,
            # the author uses positional embedding, in theory the seq can be infinite
            causal_mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()
        if self.flavor == "vanilla":
            # TODO i dont know if this is only for vanilla or can be another type
            assert attn_mask is not None
            attn_mask = (attn_mask[:, None, :] * attn_mask[:, :, None]).bool()
        out = self.sdpa(q, k, v, attn_mask, causal_mask)
        out = self.proj(out)
        return out


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
        self.vanilla_multihead = MultiHeadAttention(
            "vanilla", emb_sz, n_heads, head_sz, pdrop
        )
        self.resid_drop1 = nn.Dropout(pdrop)
        self.ln1 = nn.LayerNorm(emb_sz)
        # top part
        self.ffn = FeedForward(emb_sz, ff_sz)
        self.resid_drop2 = nn.Dropout(pdrop)
        self.ln2 = nn.LayerNorm(emb_sz)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        # NOTE: many transformer implementations now use pre-layer norm
        # i use the original paper instead, because i want to replicate
        x = self.ln1(x + self.resid_drop1(self.vanilla_multihead(x, attn_mask)))
        x = self.ln2(x + self.resid_drop2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, emb_sz: int, n_heads: int, head_sz: int, ff_sz: int, pdrop: float
    ):
        super().__init__()
        # bottom part
        self.masked_multihead = MultiHeadAttention(
            "masked", emb_sz, n_heads, head_sz, pdrop
        )
        self.resid_drop1 = nn.Dropout(pdrop)
        self.ln1 = nn.LayerNorm(emb_sz)
        # middle part
        self.cross_multihead = MultiHeadAttention(
            "cross", emb_sz, n_heads, head_sz, pdrop
        )
        self.resid_drop2 = nn.Dropout(pdrop)
        self.ln2 = nn.LayerNorm(emb_sz)
        # top part
        self.ffn = FeedForward(emb_sz, ff_sz)
        self.resid_drop3 = nn.Dropout(pdrop)
        self.ln3 = nn.LayerNorm(emb_sz)

    def forward(self, context: Tensor, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.ln1(x + self.resid_drop1(self.masked_multihead(x)))
        x = self.ln2(x + self.resid_drop2(self.cross_multihead(x, context)))
        x = self.ln3(x + self.resid_drop3(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # transformer encoder
        self.encoder_embedder = Embedder(
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
        self.decoder_embedder = Embedder(
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
        self.final_fc = nn.Linear(config.emb_sz, config.emb_sz)

    def forward(
        self,
        enc_input_ids: Tensor,
        enc_attn_mask: Tensor,
        dec_input_ids: Tensor,
        dec_attn_mask: Tensor,
    ) -> Tensor:
        encoded = self.encoder_embedder(enc_input_ids)
        for e in self.encoders:
            encoded = e(encoded, enc_attn_mask)
        decoded = self.decoder_embedder(dec_input_ids)
        for d in self.decoders:
            decoded = d(decoded, encoded, dec_attn_mask)
        decoded = self.final_fc(decoded)
        return decoded


def main():
    # TODO move all this testing shit in pytest
    # hparams
    #
    emb_sz = 256
    vocab_sz = 3333
    n_heads = 8
    head_sz = emb_sz // n_heads  # usually emb_sz == n_heads * head_sz
    pdrop = 0.1
    ff_sz = emb_sz * 4
    n_encoders = 4
    n_decoders = 5
    # 1. input hparams
    bs = 4
    seq = 18

    input_ids = torch.randint(0, vocab_sz, (bs, seq))
    embedder = Embedder(vocab_sz, emb_sz, pdrop).cuda()
    print(embedder(input_ids.cuda()).size())

    x = torch.randn(bs, seq, emb_sz)  # after the embedder part
    attn_mask = torch.randint(0, 2, (bs, seq)).bool()
    context = torch.randn(bs, seq + 1, emb_sz)  # context seq CAN be different
    vanilla_head = MultiHeadAttention("vanilla", emb_sz, n_heads, head_sz, pdrop)
    print(vanilla_head(x, attn_mask).size())
    masked_head = MultiHeadAttention("masked", emb_sz, n_heads, head_sz, pdrop)
    print(masked_head(x).size())
    cross_head = MultiHeadAttention("cross", emb_sz, n_heads, head_sz, pdrop)
    print(cross_head(x, attn_mask, context).size())

    encoder_layer = EncoderLayer(emb_sz, n_heads, head_sz, ff_sz, pdrop)
    print(encoder_layer(x).size())

    decoder_layer = DecoderLayer(emb_sz, n_heads, head_sz, ff_sz, pdrop)
    print(decoder_layer(x, context).size())

    cfg = TransformerConfig(
        n_encoders, n_decoders, vocab_sz, emb_sz, ff_sz, n_heads, head_sz, pdrop
    )
    transformer = Transformer(cfg)
    encoder_input_ids = torch.randint(0, vocab_sz, (bs, seq))
    decoder_input_ids = torch.randint(0, vocab_sz, (bs, seq + 2))
    out = transformer(encoder_input_ids, decoder_input_ids)
    print(out.size())


if __name__ == "__main__":
    main()
