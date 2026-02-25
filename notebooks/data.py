# /// script
# dependencies = [
#     "keyword-pecker",
#     "marimo",
# ]
# requires-python = ">=3.13"
#
# [tool.uv.sources]
# keyword-pecker = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    from pprint import pprint
    import polars as pl
    from transformers import AutoTokenizer
    from nltk.stem.porter import PorterStemmer

    import marimo as mo

    from keyword_pecker.data import (
        process_dataset,
        load_spacy_model,
        create_keyphrase_dataloader,
        SENTENCE_SCHEMA,
    )
    from keyword_pecker.models import KeyphraseModel
    from keyword_pecker.utils import load_config

    return (
        AutoTokenizer,
        KeyphraseModel,
        Path,
        PorterStemmer,
        SENTENCE_SCHEMA,
        create_keyphrase_dataloader,
        load_config,
        load_spacy_model,
        mo,
        pl,
        pprint,
        process_dataset,
    )


@app.cell
def _(Path):
    example_dataset = Path("../tmp/example.json")  # only has 3 abstracts
    return (example_dataset,)


@app.cell
def _(load_config):
    config = load_config("./configs/preprocessing.yaml")
    return (config,)


@app.cell
def _(AutoTokenizer, PorterStemmer, config, load_spacy_model):
    nlp = load_spacy_model(
        model_name=config["segmentation"]["spacy_model"],
        exclude=config["segmentation"]["exclude"],
    )
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["model_name"])
    stemmer = PorterStemmer()
    return nlp, stemmer, tokenizer


@app.cell
def _(example_dataset, nlp, process_dataset, stemmer, tokenizer):
    data_rows = process_dataset(
        filepath=example_dataset,
        nlp=nlp,
        spacy_batch_size=50,
        spacy_n_process=1,
        tokenizer=tokenizer,
        stemmer=stemmer,
        limit=None,
        device="cpu",
        show_progress=True,
    )
    return (data_rows,)


@app.cell
def _(SENTENCE_SCHEMA, data_rows, pl):
    example_data = pl.DataFrame(data_rows, SENTENCE_SCHEMA)
    return (example_data,)


@app.cell
def _(example_data):
    example_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example input sequence
    """)
    return


@app.cell
def _(example_data, tokenizer):
    SENTENCE_ID = 2
    print(f"Input text: {example_data.select('sent_txt')[SENTENCE_ID].item()}")
    print(
        f"Tokens: {tokenizer.decode(example_data.select('token_ids')[SENTENCE_ID].item())}"
    )
    print(
        f"Token ids: {example_data.select('token_ids')[SENTENCE_ID].item().to_list()}"
    )
    return


@app.cell
def _(create_keyphrase_dataloader, example_data):
    dataloader = create_keyphrase_dataloader(
        df=example_data,
        outer_batch_size=6,
        inner_batch_size=4,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    return (dataloader,)


@app.cell
def _(dataloader):
    print(f"Number batches: {len(dataloader)}")
    return


@app.cell
def _(dataloader):
    data_iter = iter(dataloader)
    sample_batch = next(data_iter)
    return (sample_batch,)


@app.cell
def _(pprint, sample_batch):
    pprint(sample_batch)
    return


@app.cell
def _(KeyphraseModel, load_config):
    model_cfg = load_config("./configs/models.yaml")["keyphrase_model"]
    model = KeyphraseModel(**model_cfg)
    return (model,)


@app.cell
def _(model):
    model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prepare inputs
    """)
    return


@app.cell
def _(pprint, sample_batch):
    input_ids = sample_batch["input_ids"]
    attn_mask = sample_batch["attention_mask"]

    sentence_counts = sample_batch["sentence_counts"]
    pprint(f"Input shape = {input_ids.shape}")
    pprint(f"Attn mask shape = {attn_mask.shape}")
    return attn_mask, input_ids, sentence_counts


@app.cell
def _(attn_mask, input_ids, model, sentence_counts):
    _ = model(input_ids, attn_mask, sentence_counts)
    return


if __name__ == "__main__":
    app.run()
