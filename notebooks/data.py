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
    import polars as pl
    from keyword_pecker.data import process_dataset, load_spacy_model, SENTENCE_SCHEMA
    from keyword_pecker.utils import load_config
    from transformers import AutoTokenizer
    from nltk.stem.porter import PorterStemmer

    return (
        AutoTokenizer,
        Path,
        PorterStemmer,
        SENTENCE_SCHEMA,
        load_config,
        load_spacy_model,
        pl,
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
