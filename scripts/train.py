"""Training script for the keyphrase extraction model.

Trains a KeyphraseModel (SBERT base encoder + refinement head) using the
two-level batching strategy from the original training loop:
  - Outer batch (gradient accumulation unit): outer_batch_size sentences
  - Inner batch (forward pass unit):          inner_batch_size sentences

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --device cuda --dataset kp20k
    python scripts/train.py --base-lr 2e-5 --head-lr 5e-5 --pos-weight 5.0
"""

import argparse
import math

import polars as pl
import torch
import tqdm

from keyword_pecker.data import create_keyphrase_dataloader, get_derived_path
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.utils.config import load_config


def train(
    model: KeyphraseModel,
    train_df: pl.DataFrame,
    *,
    epochs: int,
    outer_batch_size: int,
    inner_batch_size: int,
    base_lr: float,
    head_lr: float,
    pos_weight: float,
    log_every: int,
    device: torch.device,
) -> None:
    """Full training routine: optimizer, loss, dataloader creation, and epoch loop.

    Replicates the two-level batching semantics of the original training loop
    (tmp/train-model.py lines 209-305):
      - Gradients accumulate across inner batches within each outer chunk
      - optimizer.step() is called once per outer chunk (every outer_batch_size sentences)
      - The dataloader is re-created each epoch to re-shuffle at document level
        (equivalent to the legacy train_df.sample(frac=1.0))

    Args:
        model: KeyphraseModel to train (already moved to device).
        train_df: Polars DataFrame with columns [doc_id, doc_row, sent_num,
            token_ids, keyword_mask], sorted by (doc_id, doc_row, sent_num).
        epochs: Number of full passes over the training data.
        outer_batch_size: Sentences per gradient accumulation unit.
        inner_batch_size: Sentences per forward pass.
        base_lr: Learning rate for base encoder parameters.
        head_lr: Learning rate for refinement head parameters.
        pos_weight: Positive class weight for BCEWithLogitsLoss.
        log_every: Print loss every this many optimizer steps.
        device: Device to run on.
    """
    optimizer = torch.optim.AdamW(
        model.get_param_groups(base_lr=base_lr, head_lr=head_lr)
    )

    # reduction="none" so we can apply the attention mask and compute a masked
    # mean: (loss * attn_mask).sum() / attn_mask.sum()
    loss_fn = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([[pos_weight]], device=device),
    )

    inner_per_outer = math.ceil(outer_batch_size / inner_batch_size)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Re-create each epoch to re-shuffle documents (mirrors legacy .sample(frac=1.0))
        train_loader = create_keyphrase_dataloader(
            df=train_df,
            outer_batch_size=outer_batch_size,
            inner_batch_size=inner_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        model.train()
        optimizer.zero_grad()

        outer_step = 0
        inner_count = 0
        last_loss: float | None = None

        total_outer = math.ceil(len(train_loader) / inner_per_outer)
        progress = tqdm.trange(total_outer, desc="steps")

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            sentence_counts = batch["sentence_counts"]

            logits = model(input_ids, attention_mask, sentence_counts)

            # Masked mean loss: average only over non-padding token positions
            loss = loss_fn(logits, target) * attention_mask
            loss = loss.sum() / attention_mask.sum()
            loss.backward()

            last_loss = loss.detach().cpu().item()

            # Eagerly free activations (mirrors legacy explicit None assignments)
            del logits, loss, input_ids, attention_mask, target

            inner_count += 1

            if inner_count == inner_per_outer:
                optimizer.step()
                optimizer.zero_grad()

                if outer_step % log_every == 0 and last_loss is not None:
                    print(last_loss)

                outer_step += 1
                inner_count = 0
                progress.update(1)

        # Step on any remaining accumulated gradients (last incomplete outer chunk)
        if inner_count > 0:
            optimizer.step()
            optimizer.zero_grad()

            if outer_step % log_every == 0 and last_loss is not None:
                print(last_loss)

            progress.update(1)

        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the keyphrase extraction model."
    )
    parser.add_argument(
        "--dataset",
        default="kp20k",
        help="Dataset name (default: kp20k)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--outer-batch-size",
        type=int,
        default=140,
        help="Sentences per gradient accumulation unit (default: 140)",
    )
    parser.add_argument(
        "--inner-batch-size",
        type=int,
        default=70,
        help="Sentences per forward pass (default: 70)",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=2e-5,
        help="Learning rate for base encoder parameters (default: 2e-5)",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=5e-5,
        help="Learning rate for refinement head parameters (default: 5e-5)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=5.0,
        help="Positive class weight for BCEWithLogitsLoss (default: 5.0)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print loss every N optimizer steps (default: 1000)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to train on: cpu, cuda, mps (default: cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Load model configuration
    config = load_config("configs/models.yaml")
    model_config = config["keyphrase_model"]

    train_df = pl.read_parquet(get_derived_path(args.dataset, "train"))
    model = KeyphraseModel(**model_config).to(device)

    train(
        model,
        train_df,
        epochs=args.epochs,
        outer_batch_size=args.outer_batch_size,
        inner_batch_size=args.inner_batch_size,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        pos_weight=args.pos_weight,
        log_every=args.log_every,
        device=device,
    )


if __name__ == "__main__":
    main()
