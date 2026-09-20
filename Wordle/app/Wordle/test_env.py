"""Local tests for the Wordle RFT environment (no AWS calls)."""
import warnings
warnings.filterwarnings("ignore")

import main
from main import WordleGame, score_guess, parse_task, REWARD, REWARD_SCALE, WORD_ENTROPY, ALLOWED_WORDS
import json

# ── data assets loaded ──────────────────────────────────────────────────────
assert len(main.ANSWER_WORDS) > 2000, len(main.ANSWER_WORDS)
assert len(ALLOWED_WORDS) > 10000, len(ALLOWED_WORDS)
assert len(WORD_ENTROPY) > 10000 and WORD_ENTROPY.get("SOARE", 0) > 5, "entropy data"

# ── scoring incl. duplicates ────────────────────────────────────────────────
assert score_guess("CRANE", "CRANE") == ["G"] * 5
assert score_guess("SPEED", "ABIDE") == ["X", "X", "Y", "X", "Y"]  # one E yellow only

# ── win pays the outcome reward ─────────────────────────────────────────────
g = WordleGame("BEACH")
g.guess("CRANE")
r = g.guess("BEACH")
assert g.solved and "solved" in r
assert g.final_reward() > 1.0, g.final_reward()

# ── outcome ordering: faster solve > slower solve > loss > never played ──────
def _play(answer, words):
    game = WordleGame(answer)
    for w in words:
        game.guess(w)
    return game.final_reward()

fast = _play("CIGAR", ["CIGAR"])                                  # turn 1
mid = _play("CIGAR", ["SLATE", "CAIRN", "CIGAR"])                  # turn 3
slow = _play("GODLY", ["SLATE", "CLING", "GLORY", "GLOOM", "GOWLY", "GODLY"])
lost = _play("GODLY", ["SLATE", "CLING", "GLORY", "GLOOM", "GOWLY", "GOLLY"])
never = WordleGame("CIGAR").final_reward()
assert fast > mid > slow > 0 > lost > never, (fast, mid, slow, lost, never)
assert lost < 0, f"a lost game must not pay positive reward: {lost}"
assert never == main.NO_PLAY_REWARD, never

# shaping must still vary within a tier, or GRPO has no gradient
a = _play("CIGAR", ["SLATE", "CAIRN", "CIGAR"])
b = _play("CIGAR", ["FUZZY", "CAIRN", "CIGAR"])
assert a != b, "shaping must differentiate rollouts inside the same outcome tier"

# ── clue violations are penalized ───────────────────────────────────────────
g2 = WordleGame("BEACH")
g2.guess("BLIMP")   # B green at 0; L,I,M,P gray
before = g2.game_score
g2.guess("LIMPS")   # violates: no B at 0 (green), uses 4 known grays
assert g2.game_score < before - 80, g2.game_score  # 30 (green) + 4x20 (gray) - base

# ── repetition penalized, dictionary miss penalized ─────────────────────────
g3 = WordleGame("BEACH")
g3.guess("CRANE")
s = g3.game_score
g3.guess("CRANE")
assert g3.game_score < s - 30
g4 = WordleGame("BEACH")
g4.guess("XYZZY")   # 5 letters but not a word
assert g4.game_score < 0

# ── malformed input: penalized but no guess consumed ────────────────────────
g5 = WordleGame("BEACH")
out = g5.guess("hi")
assert "Invalid" in out and len(g5.guesses) == 0

# ── entropy bonus makes a good opener beat a bad one ────────────────────────
ga, gb = WordleGame("BEACH"), WordleGame("BEACH")
ga.guess("SOARE")   # top-tier opener
gb.guess("FUZZY")   # poor opener
assert ga.game_score > gb.game_score

# ── knowledge summary is plain English ──────────────────────────────────────
g6 = WordleGame("BEACH")
msg = g6.guess("CABLE")
assert "Current Knowledge" in msg and "Green" in msg and "Already Guessed" in msg

# ── no-guess rollout is worst outcome ───────────────────────────────────────
g7 = WordleGame("BEACH")
assert g7.final_reward() == main.NO_PLAY_REWARD

# ── parse_task round-trip against the real datasets ─────────────────────────
# Every row must parse and yield a playable secret. Don't pin a specific word:
# make_dataset.py reshuffles the answers whenever the datasets are regenerated.
for path in ("../../../training-data.jsonl", "../../../validation-data.jsonl"):
    answers = set()
    for line in open(path):
        prompt, answer = parse_task(json.loads(line)["prompt"])
        assert prompt, "task must carry a user prompt"
        assert len(answer) == 5 and answer.isalpha(), answer
        assert answer in ALLOWED_WORDS, f"{answer} not a legal guess"
        answers.add(answer)
    assert len(answers) > 1, f"{path} has no answer variety"

print("all environment + reward tests passed")
