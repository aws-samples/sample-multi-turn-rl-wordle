#!/usr/bin/env python3
"""Build the Wordle answer list, allowed-guess list, and entropy table from SCOWL.

Every word comes from two permissively licensed sources by Kevin Atkinson
(license text in THIRD-PARTY-LICENSES):

  SCOWL 2020.12.07  Spell Checker Oriented Word Lists. Words are grouped into
                    size levels by how common they are (10 = most common).
  AGID 2016.01.19   Automatically Generated Inflection Database. Used only to
                    drop plurals, past tenses, and other inflected forms from
                    the answer list, as Wordle does.

Both archives are downloaded once into .cache/wordlists/ and verified against
pinned SHA-256 checksums, so the output is reproducible byte for byte.

Rules:
  answers   five lowercase ASCII letters, SCOWL english/american words at size
            <= ANSWER_MAX_LEVEL, never a plural or -s verb form, not another
            inflection (past tense, -ing, -er/-est) unless it is also a
            headword in its own right (FOUND, LOWER), not offensive or profane,
            not excluded in excluded_answers.txt (manual review).
  guesses   five lowercase ASCII letters, SCOWL english/american words at size
            <= GUESS_MAX_LEVEL (inflections allowed), not offensive or
            profane, not a hashed (slur) entry in excluded_answers.txt, plus
            every answer.
  entropy   for each allowed guess, Shannon entropy (bits) of the feedback
            pattern distribution over the answer list -- the same definition
            wordle-lora-rl used, so the reward's opening-guess bonus keeps its
            meaning.

Usage:
    uv run python build_word_lists.py
    uv run python build_word_lists.py --hash WORD   # line to add a hashed exclusion
"""
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "wordlists"
OUT = ROOT / "Wordle" / "app" / "Wordle" / "data"
EXCLUDED = ROOT / "excluded_answers.txt"

SCOWL = {
    "url": "https://downloads.sourceforge.net/project/wordlist/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz",
    "sha256": "5587667caa20c4891390c2d42dbb4d5c4c3f41bee77af1457ece3ba23fb859cc",
    "prefix": "scowl-2020.12.07/",
}
AGID = {
    "url": "https://downloads.sourceforge.net/project/wordlist/AGID/2016.01.19/agid-2016.01.19.tar.gz",
    "sha256": "15d2d792d309d2dc838bf75d8abcd3feb36708c219dc5158d4fff70b89e601d1",
    "prefix": "agid-2016.01.19/",
}

SCOWL_LEVELS = (10, 20, 35, 40, 50, 55, 60, 70, 80, 95)
ANSWER_MAX_LEVEL = 35   # SCOWL "small": common everyday words
GUESS_MAX_LEVEL = 80    # SCOWL "huge": ~11k words, close to Wordle's permissiveness
CATEGORIES = ("english-words", "american-words")
TABOO = ("misc/offensive.1", "misc/offensive.2", "misc/profane.1", "misc/profane.3")
FIVE = re.compile(r"^[a-z]{5}$")


def fetch(spec: dict) -> tarfile.TarFile:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / spec["url"].rsplit("/", 1)[1]
    if not path.exists():
        print(f"downloading {spec['url']}")
        with urllib.request.urlopen(spec["url"]) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise SystemExit(f"checksum mismatch for {path.name}: {digest}")
    return tarfile.open(path)


def read_member(tar: tarfile.TarFile, prefix: str, name: str) -> list[str]:
    data = tar.extractfile(prefix + name).read()
    return data.decode("latin-1").splitlines()   # SCOWL files are ISO-8859-1


def five_letter(lines) -> set[str]:
    return {w.strip() for w in lines if FIVE.match(w.strip())}


def scowl_words(tar, max_level: int) -> set[str]:
    words = set()
    for level in (lv for lv in SCOWL_LEVELS if lv <= max_level):
        for cat in CATEGORIES:
            words |= five_letter(read_member(tar, SCOWL["prefix"], f"final/{cat}.{level}"))
    return words


def inflections(tar) -> tuple[set[str], set[str]]:
    """Split AGID inflected forms into (plural / -s forms, other inflections).

    AGID lines look like  WORD POS: form | form | ...  where noun groups are
    plurals, a verb's last group is its -s form, and the remaining verb and
    adjective groups are past tense, -ing, -er, and -est forms. Forms tagged
    "~" (only a slight chance of being an inflection) are ignored, and the
    special verbs "be" and "wit" are skipped.
    """
    plural_s, other, headwords = set(), set(), set()
    for line in read_member(tar, AGID["prefix"], "infl.txt"):
        head, _, rest = line.partition(": ")
        base, pos = head.split(" ")[0], head.split(" ")[1].rstrip("?")
        headwords.add(base)
        if base in ("be", "wit"):
            continue
        groups = rest.split(" | ")
        for gi, group in enumerate(groups):
            is_s_form = pos == "N" or (pos == "V" and gi == len(groups) - 1)
            for entry in group.split(", "):
                token = entry.split(" ")[0]
                if "~" in token:
                    continue
                word = token.rstrip("<!?")
                if word != base:
                    (plural_s if is_s_form else other).add(word)
    return plural_s, other - headwords


def word_hash(word: str) -> str:
    return hashlib.sha256(word.lower().encode()).hexdigest()


def load_excluded() -> tuple[set[str], set[str]]:
    """Return (plain words excluded from answers, hashes excluded everywhere)."""
    plain, hashed = set(), set()
    if EXCLUDED.exists():
        for ln in EXCLUDED.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("sha256:"):
                hashed.add(ln[len("sha256:"):].lower())
            else:
                plain.add(ln.lower())
    return plain, hashed


# ── entropy ─────────────────────────────────────────────────────────────────
def encode(words) -> np.ndarray:
    return np.array([[ord(c) - 65 for c in w] for w in words], dtype=np.int8)


def patterns(guess: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """Feedback codes (base-3: 0 gray, 1 yellow, 2 green) for one guess vs all
    answers, with standard duplicate-letter handling (matches main.score_guess)."""
    n = answers.shape[0]
    green = answers == guess
    remaining = np.zeros((n, 26), dtype=np.int8)
    for j in range(5):
        np.add.at(remaining, (np.nonzero(~green[:, j])[0], answers[~green[:, j], j]), 1)
    marks = np.where(green, 2, 0).astype(np.int8)
    rows = np.arange(n)
    for j in range(5):
        letter = guess[j]
        can = (~green[:, j]) & (remaining[:, letter] > 0)
        marks[can, j] = 1
        remaining[rows[can], letter] -= 1
    return marks @ (3 ** np.arange(5))


def entropy_table(guesses, answers) -> dict[str, float]:
    ans = encode(answers)
    out = {}
    for g in guesses:
        _, counts = np.unique(patterns(encode([g])[0], ans), return_counts=True)
        p = counts / len(answers)
        out[g] = float(-(p * np.log2(p)).sum())
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> None:
    scowl, agid = fetch(SCOWL), fetch(AGID)
    taboo = set()
    for name in TABOO:
        taboo |= {w.strip().lower() for w in read_member(scowl, SCOWL["prefix"], name)}
    plural_s, other_infl = inflections(agid)
    infl = plural_s | other_infl
    plain_excl, hashed_excl = load_excluded()

    def hashed_out(words: set[str]) -> set[str]:
        return {w for w in words if word_hash(w) in hashed_excl}

    common = scowl_words(scowl, ANSWER_MAX_LEVEL)
    excluded = plain_excl | hashed_out(common)
    answers = sorted(w.upper() for w in common - infl - taboo - excluded)
    all_guesses = scowl_words(scowl, GUESS_MAX_LEVEL)
    guesses = sorted((all_guesses - taboo - hashed_out(all_guesses)) | {w.lower() for w in answers})
    guesses = [w.upper() for w in guesses]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers.txt").write_text("\n".join(answers) + "\n")
    (OUT / "allowed_guesses.txt").write_text("\n".join(guesses) + "\n")
    ent = entropy_table(guesses, answers)
    (OUT / "word_entropy.json").write_text(json.dumps(ent, indent=2) + "\n")

    print(f"SCOWL <= {ANSWER_MAX_LEVEL}: {len(common)} five-letter words; "
          f"dropped {len(common & infl)} inflected, {len(common & taboo)} taboo, "
          f"{len(common & excluded)} manually excluded "
          f"({len(hashed_out(all_guesses))} hashed entries also removed from guesses)")
    print(f"answers: {len(answers)}  allowed guesses: {len(guesses)}")
    top = list(ent.items())[:5]
    print("top openers:", ", ".join(f"{w} {v:.3f}" for w, v in top))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--hash":
        print(f"sha256:{word_hash(sys.argv[2])}")
    else:
        main()
