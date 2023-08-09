# Transformer

## Overview
* The core idea of (encoder-decoder) transformer is sequence transduction. Simply, a transformer converts a sequence to another sequence that might differ in length.
* 

## Encoder stacks
* 

## Decoder stacks
* 

## Hugging Face Tokenizer
![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline.svg)
The flow of string tokenization:
* Normalization: lowercasing, unicode normalization, etc.
* Pre-tokenization: "easy" splitting based on simple rules, e.g. by whitespace, by digits, by delimiters.
* Model: split by constructing vocabulary and splitting rules based on corpus, e.g. BPE, WordPiece.
* Post-processing: adding special tokens

The flow when decoding:
* Decoder: some tokenizers need to delete some symbols when reconstructing the original sentence. The symbol (like `##`) is used to tell how to glue some tokens together.

# Usage
Create new tokenizer because we are training model from scratch with custom data
```bash
python src/transformer/make_tokenizer.py --n_vocab 30000
```

# Results
kawkawkaw

# TODO NOW
* ditch tokenizers, just use preexisting and train on my own corpus
    * see BART and T5 tokenizers (and BPE tokenizer used with NMT)
    * check how they handle processing for NMT. Is that using pair?
    * transformers paper uses BPE, i want to do that, but not the end of the world 

# TODO
* think about pretraining. Do i need that?
* write core idea 
* LATER: fix old ass yolo readme and alexnet too

# Implementation details
* memers

# References
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Karpathy's Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [Unicode normalization](https://towardsdatascience.com/what-on-earth-is-unicode-normalization-56c005c55ad0)

* [Transformer from scratch](https://www.youtube.com/watch?v=U0s0f995w14)
* [StatQuest Transformer](https://www.youtube.com/watch?v=zxQyTK8quyY)