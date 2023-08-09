from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ds_utils import load_opus_en_id


def make_corpus(train_ds: Dataset):
    sentences = []
    for batch in tqdm(train_ds.iter(100)):
        sentences += batch["en"]
        sentences += batch["id"]
    return sentences


def main(args):
    # see: https://huggingface.co/learn/nlp-course/chapter6/2
    # why use BART tokenizer as reference:
    # * i dont have to create from scratch, just change the corpus
    # * BPE, same as attention is all you need paper
    # * BART is used as seq2seq, which is the same task as paper
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    opus = load_opus_en_id()
    train_corpus = make_corpus(opus["train"])
    new_tokenizer = tokenizer.train_new_from_iterator(
        train_corpus, vocab_size=args.n_vocab
    )
    save_dir = args.dir / f"bart_bpe_opus_en_id_{args.n_vocab}"
    new_tokenizer.save_pretrained(save_dir)
    print("tokenizer saved to", save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("src/transformer/pretrained_tokenizers/"),
        help="directory to store all tokenizers",
    )
    parser.add_argument(
        "--n_vocab",
        type=int,
        default=30000,
        help="final vocab (n merges), including special tokens",
    )
    args = parser.parse_args()
    print(args)
    args.dir.mkdir(parents=True, exist_ok=True)
    main(args)
