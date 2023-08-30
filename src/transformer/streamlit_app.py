import streamlit as st
import torch
from transformers import BartTokenizer

from model import Transformer, TransformerConfig
from utils import translate


@st.cache_resource
def load_model():
    lit_ckpt_path = (
        "src/transformer/lightning_logs/version_6/checkpoints/epoch=14-step=116382.ckpt"
    )
    tokenizer_path = "src/transformer/pretrained_tokenizers/bart_bpe_opus_en_id_30000"
    # load model from pytorch lightning ckpt
    ckpt = torch.load(lit_ckpt_path, map_location="cpu")
    model_state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items()}
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
    model = Transformer(transformer_cfg).cuda().eval()
    model.load_state_dict(model_state_dict)
    # load tokenizer
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def main():
    st.title("google translate")
    st.text("en->id")
    model, tokenizer = load_model()
    col1, col2 = st.columns(2)
    with col1:
        txt = st.text_input("English", "", placeholder="Type here")
    with col2:
        result = translate(model, tokenizer, max_gen_length=128, text=[txt], t=0.2)[0]
        # dumb safeguard
        result = "" if txt == "" else result
        st.text_input("Indonesian", result)


if __name__ == "__main__":
    main()
