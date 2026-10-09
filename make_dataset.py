#!/usr/bin/env python3
"""Generate the Wordle MTRL prompt datasets from the SCOWL-derived answer list.

The RFT service reads only the `prompt` column and passes its string value to
the agent verbatim (see model-customize-mtrl-assets.html), so each row packs
the whole task -- including the secret answer -- into that one JSON string.
`parse_task` in Wordle/app/Wordle/main.py unpacks it.

Every row gets a DISTINCT secret word: duplicate prompts give the policy the
same puzzle repeatedly and waste rollouts (the docs call out unique prompts as
a dataset best practice).

Usage:
    python make_dataset.py                 # 640 train / 128 validation
    python make_dataset.py --train 1280 --val 256

640 training rows are exactly 5 batches at the default global_batch_size of
128, so every step is a full batch and one epoch is exactly 5 steps.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANSWERS = ROOT / "Wordle" / "app" / "Wordle" / "data" / "answers.txt"
PROMPT_TEXT = "Guess the 5-letter word"
SEED = 42


def write_split(path: Path, words: list[str], id_prefix: str) -> None:
    with open(path, "w") as out:
        for i, word in enumerate(words):
            task = {
                "prompt": PROMPT_TEXT,
                "answer": word.lower(),
                "id": f"{id_prefix}_{i:04d}",
            }
            # The task JSON is itself the value of the `prompt` column.
            out.write(json.dumps({"prompt": json.dumps(task)}) + "\n")
    print(f"{path.name}: {len(words)} rows, {len(set(words))} unique answers")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Wordle MTRL datasets")
    ap.add_argument("--train", type=int, default=640, help="training rows")
    ap.add_argument("--val", type=int, default=128, help="validation rows")
    args = ap.parse_args()

    words = sorted({w.strip().upper() for w in open(ANSWERS) if w.strip()})
    total = args.train + args.val
    if total > len(words):
        ap.error(f"requested {total} rows but only {len(words)} answers available")

    random.Random(SEED).shuffle(words)          # deterministic, seed 42
    picked = words[:total]                      # sampled WITHOUT replacement
    write_split(ROOT / "training-data.jsonl", picked[: args.train], "wordle_train")
    write_split(ROOT / "validation-data.jsonl", picked[args.train :], "wordle_val")
    print("train/validation answer overlap:",
          len(set(picked[: args.train]) & set(picked[args.train :])))


if __name__ == "__main__":
    main()
