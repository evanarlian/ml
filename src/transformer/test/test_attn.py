# NOTE NOTE NOTE NOTE
# torch F scaled dot product attention quirks: https://github.com/pytorch/pytorch/issues/103749
# you cannot use is_causal with attn_mask, so the solution is to use attn_mask (boolean)
# that is the result of many mask &'ed together right? WRONG
# turns out attn_mask needs float, so i just use .float() on the bool tensor. WRONG
# simply using float will not work since attn_mask is additive, not really "masking"
# so we just add with -torch.inf right? WRONG again
# this is partially wrong though, because sometimes this will cause nans in situation
# where the whole row is all -inf, like in padding mask. *IF* you are only doing causal
# mask manually (why tho, u can use is_causal directly), then nans will not happen
# so the solution is to pass attn_mask with finite super negative number, like -1e15
# ...right?? WRONG again to, since -1e15 cannot be represented in fp16 when training

# after doing all of that, remember, masked out values are garbage and should not be
# be used, this is because if you are implementing the "masking out" with addition, the
# masked out values can differ a lot if they from a full row masking like padding mask
# the difference between pure masking and additive mask might be a lot, but not to
# worry if you don't use them


# understanding many transformer masks https://discuss.pytorch.org/t/understanding-mask-size-in-transformer-example/147655
# the standard flow for encoder they need only attn_mask,
# while decoder needs encoder attn_mask, decoder attn_mask, and causal mask
# but in nn Transformer, we can give causal mask for encoder, because i think
# torch is giving us options to do causal mask in encoder, whether we need or not
# and yeah idk if this mask is additive or really "masking"


import pytest
import torch
import torch.nn.functional as F

from model import MultiHeadAttention


@pytest.mark.parametrize(
    "emb_sz, n_heads, head_sz, bs, seq_k, seq_q",
    [
        (512, 8, 64, 10, 128, 96),
        (100, 4, 100, 2, 20, 20),
        (1024, 16, 64, 1, 256, 512),
    ],
)
def test_sdpa_vs_torch_no_mask(emb_sz, n_heads, head_sz, bs, seq_k, seq_q):
    mha = MultiHeadAttention(emb_sz=emb_sz, n_heads=n_heads, head_sz=head_sz, pdrop=0.0)
    # imagine q k v are obtained through linear module
    q = torch.randn(bs, n_heads, seq_q, head_sz)
    k = torch.randn(bs, n_heads, seq_k, head_sz)
    v = torch.randn(bs, n_heads, seq_k, head_sz)  # using seq_k is not a bug
    # in this test, my sdpa require explicit mask everytime
    mask = torch.ones(seq_q, seq_k, dtype=torch.bool)
    out = mha.sdpa(q, k, v, mask)
    out_torch = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    print((out - out_torch).abs().max())
    assert torch.allclose(out, out_torch, atol=5e-5)


@pytest.mark.parametrize(
    "emb_sz, n_heads, head_sz, bs, seq",
    [
        (512, 8, 64, 10, 128),
        (100, 4, 100, 2, 20),
        (1024, 16, 64, 1, 256),
    ],
)
def test_sdpa_vs_torch_causal_mask(emb_sz, n_heads, head_sz, bs, seq):
    mha = MultiHeadAttention(emb_sz=emb_sz, n_heads=n_heads, head_sz=head_sz, pdrop=0.0)
    # imagine q k v are obtained through linear module
    q = torch.randn(bs, n_heads, seq, head_sz)
    k = torch.randn(bs, n_heads, seq, head_sz)
    v = torch.randn(bs, n_heads, seq, head_sz)
    # in this test, my sdpa require manually made triangle mask, while in torch,
    # just set is_causal=True. In theory, causal mask should always be a square matrix
    # because it comes from the same source
    mask = mha.make_causal_mask(seq, device=q.device)
    out = mha.sdpa(q, k, v, mask)
    # NOTE in torch sdpa, is_causal is similar to masking with -inf
    # the mask is really "masking", not additive
    out_torch = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
    )
    print((out - out_torch).abs().max())
    assert torch.allclose(out, out_torch, atol=5e-5)


@pytest.mark.parametrize(
    "emb_sz, n_heads, head_sz, bs, seq_k, seq_q",
    [
        (512, 8, 64, 10, 128, 96),
        (100, 4, 100, 2, 20, 20),
        (1024, 16, 64, 1, 256, 512),
    ],
)
def test_sdpa_vs_torch_random_mask(emb_sz, n_heads, head_sz, bs, seq_k, seq_q):
    mha = MultiHeadAttention(emb_sz=emb_sz, n_heads=n_heads, head_sz=head_sz, pdrop=0.0)
    # imagine q k v are obtained through linear module
    q = torch.randn(bs, n_heads, seq_q, head_sz)
    k = torch.randn(bs, n_heads, seq_k, head_sz)
    v = torch.randn(bs, n_heads, seq_k, head_sz)  # using seq_k is not a bug
    # my sdpa require a boolean mask
    # torch sdpa require a float additive mask
    mask_bool = torch.randint(0, 2, (seq_q, seq_k), dtype=torch.bool)
    mask_additive = (~mask_bool).float() * -5e4  # invert keep mask to unkeep mask
    out = mha.sdpa(q, k, v, mask_bool)
    out_torch = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask_additive, dropout_p=0.0, is_causal=False
    )
    print((out - out_torch).abs().max())
    assert torch.allclose(out, out_torch, atol=5e-5)


@pytest.mark.parametrize(
    "emb_sz, n_heads, head_sz, bs, seq",
    [
        (512, 8, 64, 10, 128),
        (100, 4, 100, 2, 20),
        (1024, 16, 64, 1, 256),
    ],
)
def test_sdpa_vs_torch_padding_mask(emb_sz, n_heads, head_sz, bs, seq):
    mha = MultiHeadAttention(emb_sz=emb_sz, n_heads=n_heads, head_sz=head_sz, pdrop=0.0)
    # imagine q k v are obtained through linear module
    q = torch.randn(bs, n_heads, seq, head_sz)
    k = torch.randn(bs, n_heads, seq, head_sz)
    v = torch.randn(bs, n_heads, seq, head_sz)
    # create fake padding mask
    # additive mask vs masking mask differs if the whole row is padded
    # we must ignore pads during assert
    orig_pad_mask = torch.arange(seq).expand(bs, seq) // (seq // 2)
    orig_pad_mask = (1 - orig_pad_mask).bool()  # (bs, seq)
    pad_mask = orig_pad_mask[:, :, None] * orig_pad_mask[:, None, :]
    pad_mask = pad_mask.unsqueeze(-3)  # compensate for n_heads
    pad_mask_additive = (~pad_mask).float() * -5e4  # invert keep mask to unkeep mask
    # reshape out for easier testing
    out = mha.sdpa(q, k, v, pad_mask)
    out = out.transpose(-2, -3).reshape(bs, seq, n_heads * head_sz)
    out_torch = F.scaled_dot_product_attention(
        q, k, v, attn_mask=pad_mask_additive, dropout_p=0.0, is_causal=False
    )
    out_torch = out_torch.transpose(-2, -3).reshape(bs, seq, n_heads * head_sz)
    # both outs is (bs, seq, n_heads*head_sz)
    # during allclose, make sure to skip the padded part, they are garbage!
    for p_mask, o, o_t in zip(orig_pad_mask, out, out_torch):
        print((o[p_mask] - o_t[p_mask]).abs().max())
        assert torch.allclose(o[p_mask], o_t[p_mask], atol=5e-5)
