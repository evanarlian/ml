from pathlib import Path

import torch
from torch import Tensor
from transformers import BartTokenizer, PreTrainedTokenizer

from model import Transformer


def softmax_temp(logits: Tensor, t: float, dim: int):
    """Softmax with temperature."""
    return (logits / t).softmax(dim)


@torch.no_grad()
def translate(
    model: Transformer,
    tokenizer: PreTrainedTokenizer,
    max_gen_length: int,
    text: list[str],
    t: float = 1.0,
) -> list[str]:
    """
    Batch translate texts.

    Args:
        model (Transformer): Encoder decoder transformer model
        tokenizer (PreTrainedTokenizer): HF tokenizer
        max_gen_length (int): Maximum generated length, including the start token
        text (list[str]): Texts to translate
        t (float, optional): Temperature, < 1.0 for confidence, > 1.0 for creativity

    Returns:
        list[str]: Translated texts
    """
    # save previous state of model
    orig_train_state = model.training
    model.eval()

    # useful variables
    device = next(model.parameters()).device
    bs = len(text)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    ctx = tokenizer(text=text, truncation=False, padding=True, return_tensors="pt")
    ctx_input_ids = ctx["input_ids"].to(device)
    ctx_pad_mask = ctx["attention_mask"].to(device)

    # autoregressive translation
    tgt_input_ids = torch.full((bs, 1), bos, dtype=torch.long, device=device)
    tgt_pad_mask = torch.full((bs, 1), 1, dtype=torch.long, device=device)
    eos_found = torch.zeros(bs, 1, device=device).bool()
    for i in range(max_gen_length - 1):
        if eos_found.all():
            break
        # inference for new token (last token)
        logits = model(ctx_input_ids, ctx_pad_mask, tgt_input_ids, tgt_pad_mask)
        last_logits = logits[:, -1]  # (bs, vocab_sz)
        last_probs = softmax_temp(last_logits, t, -1)  # (bs, vocab_sz)
        next_tokens = last_probs.multinomial(1)
        next_tokens = torch.where(eos_found, pad, next_tokens)
        # construct decoder input for next autoregression
        tgt_input_ids = torch.cat([tgt_input_ids, next_tokens], dim=-1)
        tgt_pad_mask = torch.cat([tgt_pad_mask, (~eos_found).long()], dim=-1)
        # update eos for next round
        eos_found |= next_tokens == eos

    # batch decode
    result = [
        tgt[mask] for tgt, mask in zip(tgt_input_ids.cpu(), (tgt_pad_mask == 1).cpu())
    ]
    result = tokenizer.batch_decode(
        result, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # return back the training state
    model.train(orig_train_state)
    return result


def get_version(root_dir: Path, log_dir: str = "lightning_logs"):
    """Extract number from every version and find new max version"""
    max_ver = 0
    log_path = root_dir / log_dir
    if not log_path.exists():
        return 0
    for ver in (root_dir / log_dir).iterdir():
        try:
            max_ver = max(max_ver, int(ver.name.split("_")[-1]))
        except ValueError:
            pass
    return max_ver + 1


def main():
    # TODO pytest
    from model import Transformer, TransformerConfig

    cfg = TransformerConfig(
        n_encoders=1,
        n_decoders=1,
        vocab_sz=30000,
        emb_sz=128,
        ff_sz=128 * 4,
        n_heads=4,
        head_sz=128 // 4,
        pdrop=0.0,
    )
    model = Transformer(cfg).train().cuda()
    tokenizer_dir = "src/transformer/pretrained_tokenizers/bart_bpe_opus_en_id_30000"
    tokenizer = BartTokenizer.from_pretrained(tokenizer_dir)
    result = translate(model, tokenizer, max_gen_length=100, text=["meme", "dog"])
    print(result)


if __name__ == "__main__":
    main()
