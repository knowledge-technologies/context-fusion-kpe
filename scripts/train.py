from keyword_pecker.data import create_keyphrase_dataloader, get_derived_path
import polars as pl
from pathlib import Path

DATASET = "kp20k"
train_parquet = Path(get_derived_path(DATASET), "train.parquet")
train_df = pl.read_parquet(train_parquet)

dataloader = create_keyphrase_dataloader(
    df=train_df,
    inner_batch_size=70,
    outer_batch_size=140,
    shuffle=True,
    drop_last=False,
    num_workers=0,
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
