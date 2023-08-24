import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, PreTrainedTokenizer


def load_opus_en_id(split: str | None = None):
    opus = load_dataset("opus100", "en-id", split=split)
    # remove outer "translation key"
    opus = opus.map(lambda row: row["translation"], remove_columns="translation")
    return opus


class OpusEnId(Dataset):
    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ds = load_opus_en_id(split)

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return f"OpusEnId(split={self.split}, n={len(self)})"

    def __getitem__(self, i):
        # we need 5 items from this method:
        # * encoder_inpus_ids (tokens: <s>  hello world </s>)
        # * decoder_input_ids (tokens: <s>  halo  dunia)
        # * labels            (tokens: halo dunia </s>)
        # * encoder_attn_mask
        # * decoder_attn_mask
        # NOTE: the code is a little bit hacky, because of the shifting pattern
        # in the decoder_input_ids and label, forced us to truncate one more than the
        # max length. Also the padding is not handled here, it will be handled in the
        # collate_fn alongside the tensor conversion. What we are doing here is
        # similar to huggingface's DataCollatorFor...
        row = self.ds[i]
        ctx = self.tokenizer(
            text=row["en"], truncation=True, max_length=self.max_length
        )
        tgt = self.tokenizer(
            text_target=row["id"], truncation=True, max_length=self.max_length + 1
        )
        d = {
            "ctx_text": row["en"],
            "ctx_input_ids": ctx["input_ids"],
            "ctx_pad_mask": ctx["attention_mask"],
            "tgt_text": row["id"],
            "tgt_input_ids": tgt["input_ids"][:-1],
            "tgt_pad_mask": tgt["attention_mask"][:-1],
            "labels": tgt["input_ids"][1:],
        }
        return d

    def collate_fn(self, batch: list[dict]):
        # in collate_fn, we will do:
        # * pad enc stuff to the longest length in batch
        # * pad dec stuff to the longest length in batch
        # * pad label with -100, to ignore during crossentropyloss
        # * convert to torch tensor
        pad = self.tokenizer.pad_token_id
        d = {}
        # fmt: off
        max_c = max(len(b["ctx_input_ids"]) for b in batch)
        d["ctx_input_ids"] = [b["ctx_input_ids"] + ([pad] * (max_c-len(b["ctx_input_ids"]))) for b in batch]  # noqa: E501
        d["ctx_pad_mask"] = [b["ctx_pad_mask"] + ([0] * (max_c-len(b["ctx_pad_mask"]))) for b in batch]  # noqa: E501
        max_t = max(len(b["tgt_input_ids"]) for b in batch)
        d["tgt_input_ids"] = [b["tgt_input_ids"] + ([pad] * (max_t-len(b["tgt_input_ids"]))) for b in batch]  # noqa: E501
        d["tgt_pad_mask"] = [b["tgt_pad_mask"] + ([0] * (max_t-len(b["tgt_pad_mask"]))) for b in batch]  # noqa: E501
        d["labels"] = [b["labels"] + ([-100] * (max_t-len(b["labels"]))) for b in batch]
        d = {k: torch.tensor(v) for k, v in d.items()}
        # fmt: on
        d["ctx_text"] = [b["ctx_text"] for b in batch]
        d["tgt_text"] = [b["tgt_text"] for b in batch]
        return d

    def create_dataloader(
        self, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=drop_last,
        )


def main():
    from tqdm.auto import tqdm

    tokenizer = BartTokenizer.from_pretrained(
        "./src/transformer/pretrained_tokenizers/bart_bpe_opus_en_id_30000"
    )
    train_ds = OpusEnId(split="train", tokenizer=tokenizer, max_length=512)
    print(train_ds)
    print(train_ds[0].keys())
    dl = train_ds.create_dataloader(
        batch_size=256, shuffle=True, num_workers=6, drop_last=False
    )
    for i, batch in zip(range(100), tqdm(dl)):
        batch = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


if __name__ == "__main__":
    main()
