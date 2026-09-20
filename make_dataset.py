#!/usr/bin/env python3
"""Generate the Wordle MTRL prompt datasets from the NYT answer list.

The RFT service reads only the `prompt` column and passes its string value to
the agent verbatim (see model-customize-mtrl-assets.html), so each row packs
the whole task -- including the secret answer -- into that one JSON string.
`parse_task` in Wordle/app/Wordle/main.py unpacks it.

Every row gets a DISTINCT secret word: duplicate prompts give the policy the
same puzzle repeatedly and waste rollouts (the docs call out unique prompts as
a dataset best practice).

Usage:
    python make_dataset.py                 # 600 train / 100 validation
    python make_dataset.py --train 1000 --val 200
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANSWERS = ROOT / "Wordle" / "app" / "Wordle" / "data" / "nyt_answers_wordle_list.txt"
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
    ap.add_argument("--train", type=int, default=600, help="training rows")
    ap.add_argument("--val", type=int, default=100, help="validation rows")
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
