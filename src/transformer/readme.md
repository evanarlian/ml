# Transformer

## Overview
* The core idea of (encoder-decoder) transformer is sequence transduction. Simply, a transformer converts a sequence to another sequence that might differ in length.
* Transformer encoder uses standard self attention.
* Transformer decoder uses 2 types of attention, causal attention and cross attention.
* The target is shifted right, to create something called teacher forcing.
* Transformer with static positional encoding can, in theory, process infinite sequence length (e.g. Transformer from the OG paper). Transformer with learned positional encoding can only process finite sequence length (e.g. BERT).


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
Modify `config.py` and train the model
```bash
python src/transformer/train.py
```
Testing using pytest
```bash
PYTHONPATH=src/transformer pytest src/transformer/ -q
```

# Results
English to Indonesian translation got about 14.0 and 18.0 SacreBLEU score on test and validation set respectively. For the metrics see on [wandb](https://wandb.ai/evanarlian/transformer_mt).

See more on `translate.ipynb`, below are some of the example.
| English                          | Indonesian (t=0.3)                          | Indonesian (t=1.2)                                        |
|:---------------------------------|:--------------------------------------------|:-----------------------------------------------------------|
| The cat sat on the mat.          | Kucing duduk di atas matras.                | ketika menemui kucing itu untuk sebuah melainkan terputus. |
| My teddy bear is so cuddly.      | Beruang Bantengku sangat marah.             | itasuoLagipula begitu mintaasnya vaksin.                   |
| Let's read a fun story together. | Mari kita baca cerita menyenangkan bersama. | Ket bercerita menyenangkan bersama-sama.                   |
| The principles of genetic engineering have revolutionized the field of biotechnology. | Prinsip dari teknik genetik telah telah mencapai bidang biotenologi biote. | Prinsip dari teknik genetik telah menghubungkan jauhnya secara bi pen kendali biote detik.                                           |
| Postmodern literature often blurs the line between fiction and reality.               | Postmodern sastra sering berbaur antara fiksi dan kenyataan.               | Dia seringlo arwah dengan cahayaKhusus pamanku, katamu pesta perpisahan dengan bat Datang saat jangkaasaan menyuntikkan titik ajaib. |


# TODO
* clean todos
* pytest
* attention visualization like bertviz!!!!!!!!!!11

# Implementation details
* Original tasks in the paper are English to German and English to French translation. This project is English to Indonesian (so that I can verify the quality).
* The dataset for English-Indonesian is obtained from OPUS (1mil), while English-German (4.5mil) and English-French (36mil) are obtained from WMT 2014. This might explain the low SacreBLEU score.

# References
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Karpathy's Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [Unicode normalization](https://towardsdatascience.com/what-on-earth-is-unicode-normalization-56c005c55ad0)
