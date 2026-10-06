"""Wordle agent for SageMaker multi-turn reinforcement learning (RFT).

Environment + reward design ported from the wordle-lora-rl project
(https://github.com/charbull/wordle-lora-rl), adapted from tag-parsing GRPO
to a Strands tool-calling agent on Bedrock AgentCore:

- Real dictionary validation against the NYT word lists (data/).
- Plain-English "Current Knowledge" state summaries in tool feedback
  (wordle-lora-rl Lesson 3: structured natural language beats symbols).
- Composite reward: clue-consistency penalties (green/yellow/gray violations),
  repetition and dictionary penalties, entropy-based opening bonus,
  new-letter exploration bonus, possibility-reduction bonus, stagnation and
  time penalties. Format failure is the worst outcome (Lesson 2: the penalty
  for not playing must dominate every strategic mistake).

BedrockAgentCoreApp serves the HTTP contract (port 8080) that AgentCore
invokes during training; the @sagemaker_rft_handler decorator reports
CompleteRollout + UpdateReward from the returned {"reward": ...}.
Decorator order matters: @app.entrypoint outermost, @sagemaker_rft_handler inner.
"""

import json
import logging
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

from botocore import session as botocore_session

from bedrock_agentcore import BedrockAgentCoreApp
from sagemaker.core.token_generator import generate_token
from sagemaker.train.rft import sagemaker_rft_handler
from sagemaker.train.rft.adapters.strands import wrap_model
from strands import Agent, tool
from strands.models.openai import OpenAIModel

logger = logging.getLogger(__name__)

WORD_LENGTH = 5
MAX_GUESSES = 6
MAX_TOOL_CALLS = 12  # hard cap incl. invalid calls so a rollout can't loop forever
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ── Bearer token handling ───────────────────────────────────────────────────
# Two failure modes to avoid, both seen in real training runs:
#
#  1. generate_token() builds a NEW botocore Session and walks the credential
#     chain on every call. Under concurrent rollouts that hammers the container
#     credential provider until it returns None -> "No AWS credentials found".
#  2. Caching the resulting token for a long time goes stale, because it is
#     signed with the container's IAM role credentials, which rotate ->
#     "403 Authentication failed: Please make sure your API Key is valid",
#     which starves the trajectory of sampling and fails the whole job.
#
# Fix both: share ONE botocore Session process-wide (its Credentials object
# refreshes itself, so no provider storm and no stale signing material) and
# keep only a short token cache on top. invalidate_token() forces a re-sign
# when the endpoint rejects our key anyway.
_botocore_session = botocore_session.Session()


class _SharedSessionCredentials:
    """CredentialProvider over the shared session's auto-refreshing creds."""

    def load(self):
        return _botocore_session.get_credentials()


_TOKEN_TTL = 300.0
_TOKEN_RETRIES = 3
_token_value: str | None = None
_token_expiry = 0.0
_token_lock = threading.Lock()


def invalidate_token() -> None:
    """Drop the cached token so the next call re-signs with fresh credentials."""
    global _token_value, _token_expiry
    with _token_lock:
        _token_value, _token_expiry = None, 0.0


def _looks_like_auth_failure(err: Exception) -> bool:
    """True if the endpoint rejected our bearer token (stale/invalid)."""
    msg = str(err).lower()
    return "403" in msg or "authentication failed" in msg or "access_denied" in msg


def get_cached_token() -> str:
    """Return a bearer token for the RFT Runtime, cached across rollouts."""
    global _token_value, _token_expiry
    now = time.monotonic()
    if _token_value is not None and now < _token_expiry:
        return _token_value
    with _token_lock:
        # Another thread may have refreshed while we waited for the lock.
        now = time.monotonic()
        if _token_value is not None and now < _token_expiry:
            return _token_value
        last_err: Exception | None = None
        for attempt in range(_TOKEN_RETRIES):
            try:
                _token_value = generate_token(
                    region=AWS_REGION,
                    aws_credentials_provider=_SharedSessionCredentials(),
                )
                _token_expiry = time.monotonic() + _TOKEN_TTL
                return _token_value
            except Exception as e:  # credential chain hiccup under load
                last_err = e
                logger.warning("generate_token attempt %d/%d failed: %s",
                               attempt + 1, _TOKEN_RETRIES, e)
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"Could not generate RFT bearer token: {last_err}")

# ── Word lists & entropy (from wordle-lora-rl/data) ─────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"

with open(DATA_DIR / "nyt_answers_wordle_list.txt") as f:
    ANSWER_WORDS = [w.strip().upper() for w in f if w.strip()]
with open(DATA_DIR / "nyt_possible_wordle_list.txt") as f:
    ALLOWED_WORDS = {w.strip().upper() for w in f if w.strip()}
ALLOWED_WORDS |= set(ANSWER_WORDS)

with open(DATA_DIR / "word_entropy.json") as f:
    WORD_ENTROPY = json.load(f)  # {WORD: bits of information gain}

# ── Reward constants (wordle-lora-rl config/grpo_lora_config.json) ──────────
REWARD = {
    "solution_correct_guess": 150.0,
    "valid_guess_base": 15.0,
    "information_gain_bonus_coeff": 7.5,
    "new_letter_bonus": 2.0,
    "possibility_reduction_bonus": 15.0,
    "time_penalty_per_guess": 1.0,
    "gray_letter_penalty": 20.0,
    "yellow_letter_penalty": 20.0,
    "green_position_penalty": 30.0,
    "green_reuse_penalty": 3.0,
    "yellow_reuse_penalty": 1.5,
    "repetition_penalty": 40.0,
    "not_in_dictionary_penalty": 35.0,
    "format_fail_penalty": 200.0,
    # Charged when a guess had to be recovered from a raw-text tool call
    # instead of a structured one. Small on purpose: a properly formatted call
    # must stay strictly better, but a drifting policy still needs to see
    # reward VARIANCE, otherwise every rollout pins at the format-failure floor
    # and GRPO's advantage goes to zero with no gradient back to valid syntax.
    "text_format_penalty": 5.0,
}
# game_score above accumulates SHAPING only: clue-consistency, exploration and
# dictionary signal per guess. It is normalised by this scale before use.
REWARD_SCALE = REWARD["solution_correct_guess"]

# ── Outcome vs shaping ──────────────────────────────────────────────────────
# The rollout reward is outcome + a small shaping term:
#
#   final = outcome + SHAPING_WEIGHT * clip(game_score / REWARD_SCALE)
#
# Summing per-guess shaping straight into the reward (the original design,
# inherited from wordle-lora-rl where each guess was scored as an independent
# sample) produced two inversions in real traces:
#   * a LOST game still paid positive reward (+0.49), so the policy could farm
#     ~0.5 without ever solving;
#   * winning on turn 1 (+0.99) scored LOWER than winning on turn 3 (+1.61),
#     because five turns of +15 base plus entropy and reduction bonuses dwarfed
#     the -1/guess time penalty. The reward mildly favoured dawdling.
# Making the outcome dominant fixes both while keeping shaping as a gradient.
SOLVE_BASE_REWARD = 1.0       # any solve clears every non-solve
SOLVE_SPEED_BONUS = 0.5       # full for a first-guess solve, 0 on the last guess
LOSS_REWARD = -0.5            # played it out, never solved: must be negative
NO_PLAY_REWARD = -1.5         # never guessed: strictly the worst outcome
SHAPING_WEIGHT = 0.15         # small enough that it cannot outrank the outcome
SHAPING_CLIP = 1.0            # bound shaping so one wild game can't dominate

# ── Reasoning budget ────────────────────────────────────────────────────────
# sampling_max_tokens is capped at 8192 by the service, and a reasoning model
# like GPT-OSS-20B can spend all of it thinking before its first tool call,
# which starves the trajectory of sampling and fails the job. gpt-oss supports
# a configurable reasoning effort (low/medium/high), so ask for "low" both in
# the Harmony system message and as a request parameter.
#
# Both trainable models here are reasoning models that accept reasoning_effort:
# gpt-oss (Harmony) and Nova 2 Lite (extended thinking). Set REASONING_EFFORT
# to something outside low/medium/high (e.g. "none") only for a model that
# rejects the field; the request param is then dropped and the Harmony system
# hint omitted.
#
# Caveat from the Nova 2 troubleshooting guide: with HIGH effort you should
# leave temperature/topP/maxTokens unset so the model picks its own. We use
# "low", where that does not apply.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low").strip().lower()
_reasoning_effort_ok = REASONING_EFFORT in ("low", "medium", "high")

# Separate switch: gpt-oss additionally reads the reasoning level from a
# "Reasoning: <level>" line in the Harmony system message. That convention is
# gpt-oss-specific -- Nova takes the level from the request parameter only --
# so keep it opt-in via REASONING_PROMPT_HINT=true rather than always emitting
# a line that other models would just read as stray instruction text.
_hint_enabled = os.environ.get("REASONING_PROMPT_HINT", "false").strip().lower() in (
    "1", "true", "yes")
_REASONING_HINT = (f"Reasoning: {REASONING_EFFORT}\n\n"
                   if _reasoning_effort_ok and _hint_enabled else "")

SYSTEM_PROMPT = f"""{_REASONING_HINT}You are playing Wordle. A secret {WORD_LENGTH}-letter English word has been chosen.

Rules:
- Use the guess_word tool to submit a valid {WORD_LENGTH}-letter English word.
- You have {MAX_GUESSES} guesses total.
- After each guess the tool reports your result and a Current Knowledge summary:
  green letters (correct position), yellow letters (in the word, wrong spot),
  gray letters (not in the word), and words already guessed.

Strategy:
- Open with a word that maximizes information (many common, distinct letters).
- Every later guess MUST respect the clues: keep greens in place, include all
  yellows somewhere new, and never reuse gray letters.
- Do not repeat any words you have already guessed.
- Prefer guesses that introduce new letters to narrow the answer quickly.

CRITICAL - how to respond:
- Call guess_word IMMEDIATELY. Do not deliberate before your first guess:
  pick a strong opener and submit it right away.
- Keep any reasoning to one short sentence per turn. Long reasoning wastes
  your token budget and forfeits the game.
- Never explain your full strategy, enumerate candidate words, or think step
  by step at length. Just guess, read the feedback, and guess again.
- Stop once you solve it or run out of guesses, then state the outcome in one
  short sentence."""

# Used when a previous attempt burned its whole token budget without guessing.
RETRY_PROMPT = ("Call the guess_word tool right now with any common 5-letter "
                "word. Do not write any reasoning first.")

# Retries exist only to rescue a rollout that never played a guess, which would
# otherwise fail the whole job. They are bounded twice over: by attempt count
# and by wall clock. Three full 6-guess conversations could outlast the
# service's reward window, which fails the job with "The rollout completed but
# no reward was received in time" -- so stop retrying once the budget is spent
# and report whatever score we have. Keep the budget well inside the
# rollout_timeout hyperparameter (600s by default).
AGENT_MAX_ATTEMPTS = 2
ROLLOUT_BUDGET_SECONDS = float(os.environ.get("ROLLOUT_BUDGET_SECONDS", "420"))

# Strands splats params into chat.completions.create() and wrap_model merges
# rather than replaces them, so reasoning_effort survives RFT injection. If the
# endpoint rejects the unknown field anyway, disable it process-wide.
def _looks_like_param_rejection(err: Exception) -> bool:
    """True if the endpoint rejected the reasoning_effort field itself."""
    msg = str(err).lower()
    return "reasoning_effort" in msg or (
        "400" in msg and any(s in msg for s in
                             ("unknown", "unexpected", "unrecognized", "extra field"))
    )


def score_guess(guess: str, answer: str) -> list[str]:
    """Standard Wordle marks G/Y/X with correct duplicate-letter handling."""
    marks = ["X"] * WORD_LENGTH
    remaining: Counter = Counter()
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            marks[i] = "G"
        else:
            remaining[a] += 1
    for i, g in enumerate(guess):
        if marks[i] != "G" and remaining[g] > 0:
            marks[i] = "Y"
            remaining[g] -= 1
    return marks


class WordleGame:
    """Per-rollout Wordle environment with wordle-lora-rl's clue tracking
    and composite reward. Accumulates a game score across guesses."""

    def __init__(self, answer: str):
        self.answer = answer.upper()
        self.guesses: list[str] = []
        self.fallback_guesses = 0
        self.tool_calls = 0
        self.solved = False
        self.game_score = 0.0
        # Clue state (green truth dominates yellow/gray, per wordle-lora-rl)
        self.known_green: dict[int, str] = {}      # {index: letter}
        self.known_yellow: Counter = Counter()      # {letter: min required count}
        self.yellow_positions: dict[str, set] = {}  # {letter: positions tried}
        self.known_gray: set[str] = set()
        self.letters_seen: set[str] = set()

    # ── state ────────────────────────────────────────────────────────────
    @property
    def over(self) -> bool:
        return self.solved or len(self.guesses) >= MAX_GUESSES

    def _update_knowledge(self, guess: str, marks: list[str]) -> None:
        in_secret_this_turn: Counter = Counter()
        for i, m in enumerate(marks):
            if m in ("G", "Y"):
                in_secret_this_turn[guess[i]] += 1
        for letter, count in in_secret_this_turn.items():
            self.known_yellow[letter] = max(self.known_yellow[letter], count)
        for i, m in enumerate(marks):
            letter = guess[i]
            if m == "G":
                self.known_green[i] = letter
            elif m == "Y":
                self.yellow_positions.setdefault(letter, set()).add(i)
            elif m == "X" and in_secret_this_turn[letter] == 0:
                self.known_gray.add(letter)
        # Green is the highest truth: drop from yellow/gray constraint sets.
        for letter in set(self.known_green.values()):
            self.known_yellow.pop(letter, None)
            self.known_gray.discard(letter)
        self.letters_seen.update(guess)

    def possibilities(self) -> list[str]:
        """Answers still consistent with all clues (wordle-lora-rl
        find_valid_completions)."""
        valid = []
        for word in ANSWER_WORDS:
            counts = Counter(word)
            if any(word[i] != l for i, l in self.known_green.items()):
                continue
            if any(l in counts for l in self.known_gray):
                continue
            if any(counts[l] < c for l, c in self.known_yellow.items()):
                continue
            if any(word[p] == l for l, ps in self.yellow_positions.items() for p in ps):
                continue
            valid.append(word)
        return valid

    def knowledge_summary(self) -> str:
        """Plain-English state summary (wordle-lora-rl Lesson 3)."""
        green = ["_"] * WORD_LENGTH
        for i, l in self.known_green.items():
            green[i] = l
        lines = ["Current Knowledge:",
                 f"- Correct Position (Green): {' '.join(green)}"]
        if self.known_yellow:
            yellow = ", ".join(f"'{l}' (at least {c})" for l, c in sorted(self.known_yellow.items()))
            lines.append(f"- In Word, Wrong Position (Yellow): {yellow}")
        else:
            lines.append("- In Word, Wrong Position (Yellow): None")
        gray = ", ".join(sorted(self.known_gray)) if self.known_gray else "None"
        lines.append(f"- Not in Word (Gray): {gray}")
        lines.append(f"- Words Already Guessed: {', '.join(self.guesses) or 'None'}")
        lines.append(f"- Guesses Remaining: {MAX_GUESSES - len(self.guesses)}")
        return "\n".join(lines)

    # ── reward components (ported from wordle-lora-rl rewards.py) ─────────
    def _violation_penalty(self, guess: str) -> float:
        penalty = 0.0
        for i, l in self.known_green.items():
            if guess[i] != l:
                penalty += REWARD["green_position_penalty"]
        counts = Counter(guess)
        for l, required in self.known_yellow.items():
            if counts[l] < required:
                penalty += REWARD["yellow_letter_penalty"]
        for l in set(guess):
            if l in self.known_gray:
                penalty += REWARD["gray_letter_penalty"]
        return penalty

    def _stagnation_penalty(self, guess: str) -> float:
        penalty = 0.0
        for i, l in self.known_green.items():
            if guess[i] == l:
                penalty += REWARD["green_reuse_penalty"]
        for l in set(guess):
            if l in self.known_yellow:
                penalty += REWARD["yellow_reuse_penalty"]
        return penalty

    def _strategic_bonus(self, guess: str) -> float:
        if not self.guesses:  # turn 1: pre-computed information gain
            return WORD_ENTROPY.get(guess, 0.0) * REWARD["information_gain_bonus_coeff"]
        new_letters = set(guess) - self.letters_seen
        return len(new_letters) * REWARD["new_letter_bonus"]

    # ── the tool action ────────────────────────────────────────────────────
    def guess(self, word: str) -> str:
        self.tool_calls += 1
        if self.tool_calls > MAX_TOOL_CALLS:
            return "Game over: too many tool calls. Stop guessing."
        if self.over:
            return ("Game over: you already solved it." if self.solved
                    else f"Game over: all {MAX_GUESSES} guesses used. Stop guessing.")

        word = word.strip().upper()
        if len(word) != WORD_LENGTH or not word.isalpha():
            # Malformed input: penalize, don't consume a guess.
            self.game_score -= REWARD["format_fail_penalty"] / 4
            return (f"Invalid guess '{word}': must be exactly {WORD_LENGTH} letters "
                    "(a-z only). This did not use up a guess. Try again.")

        if word in self.guesses:
            self.game_score -= REWARD["repetition_penalty"]
            self.game_score -= REWARD["time_penalty_per_guess"]
            self.guesses.append(word)
            return (f"You already guessed {word} (repetition penalty). "
                    f"{self.knowledge_summary()}")

        # Score the guess BEFORE updating knowledge (violations are judged
        # against what was known when the guess was made).
        marks = score_guess(word, self.answer)
        possibilities_before = len(self.possibilities()) if self.guesses else 0

        if word == self.answer:
            # The solve bonus lives in the OUTCOME term (see final_reward), not
            # in the shaping accumulator, so speed can be rewarded properly.
            self.game_score -= REWARD["time_penalty_per_guess"]
            self.guesses.append(word)
            self.solved = True
            return (f"Guess {len(self.guesses)}/{MAX_GUESSES}: {word} — all green. "
                    "Correct, you solved it!")

        score = REWARD["valid_guess_base"]
        score -= self._violation_penalty(word)
        score -= self._stagnation_penalty(word)
        score += self._strategic_bonus(word)
        if word not in ALLOWED_WORDS:
            score -= REWARD["not_in_dictionary_penalty"]

        self._update_knowledge(word, marks)
        self.guesses.append(word)

        # Possibility-reduction bonus (not applicable on the first guess)
        if possibilities_before > 0:
            after = len(self.possibilities())
            reduction = (possibilities_before - after) / possibilities_before
            score += reduction * REWARD["possibility_reduction_bonus"]

        score -= REWARD["time_penalty_per_guess"]
        self.game_score += score

        marked = ", ".join(f"{l}={'green' if m == 'G' else 'yellow' if m == 'Y' else 'gray'}"
                           for l, m in zip(word, marks))
        result = f"Guess {len(self.guesses)}/{MAX_GUESSES}: {word} -> {marked}\n{self.knowledge_summary()}"
        if not self.solved and len(self.guesses) >= MAX_GUESSES:
            result += f"\nOut of guesses — the word was '{self.answer}'."
        return result

    def guess_from_text(self, word: str) -> str:
        """Play a guess recovered from a raw-text tool call.

        Scored like a normal guess minus a formatting penalty. RL fine-tuning
        can drift the policy into emitting tool syntax as plain text; without
        this recovery path such a rollout plays nothing, every rollout in the
        GRPO group scores the same format-failure floor, the advantage is zero
        and the policy can never learn its way back.
        """
        self.fallback_guesses += 1
        result = self.guess(word)
        self.game_score -= REWARD["text_format_penalty"]
        return result

    def final_reward(self) -> float:
        """Rollout reward: a dominant outcome term plus bounded shaping.

        Guaranteed by construction: any solve > any loss > never played.
        Within solves, each guess saved adds 0.1 to the outcome term, but
        shaping (up to +/-0.15) can reorder two solves one guess apart.
        Shaping can never make a loss look like a win, but it still varies between rollouts -- which is
        what GRPO needs to compute a non-zero advantage.
        """
        if not self.guesses:
            # No guess ever executed. Worst outcome, and deliberately below any
            # played-out game so a format failure can never look attractive.
            return NO_PLAY_REWARD

        shaping = self.game_score / REWARD_SCALE
        shaping = max(-SHAPING_CLIP, min(SHAPING_CLIP, shaping))

        if self.solved:
            # 1.0 for solving on the final guess, up to 1.5 for a first-guess
            # solve, so fewer guesses earn a larger outcome term.
            speed = (MAX_GUESSES - len(self.guesses)) / (MAX_GUESSES - 1)
            outcome = SOLVE_BASE_REWARD + SOLVE_SPEED_BONUS * speed
        else:
            outcome = LOSS_REWARD

        return outcome + SHAPING_WEIGHT * shaping


# ── Raw-text tool-call recovery ─────────────────────────────────────────────
# Observed in a real Nova run after ~15 steps of RL: the policy stopped
# emitting structured tool calls and started writing the syntax as plain text,
# e.g.
#     <tools>
#     <__function=guess_word>
#     <__parameter=guess>STARE</__parameter>
#     </__function>
#     </tools>
# (newline- and space-separated variants, mixed case). Strands never executes
# these, so the rollout plays nothing. Recover the word so the game still runs.
_TEXT_GUESS_PATTERNS = (
    # The observed Nova drift format.
    re.compile(r"__parameter\s*=\s*guess\s*>\s*([A-Za-z]{5})\s*<", re.IGNORECASE),
    # JSON-ish leakage, e.g. {"guess": "STARE"} or guess=STARE.
    re.compile(r"[\"']?guess[\"']?\s*[:=]\s*[\"']?([A-Za-z]{5})\b", re.IGNORECASE),
    # Python/JS-style call written as prose, e.g. guess_word("crane").
    # Gemma-3 on Bedrock emits this instead of a structured tool call.
    re.compile(r"guess_word\s*\(\s*[\"']?([A-Za-z]{5})[\"']?\s*\)", re.IGNORECASE),
)


def extract_text_guesses(messages) -> list[str]:
    """Pull guesses out of assistant text blocks, in order, de-duplicated."""
    found: list[str] = []
    for msg in messages or []:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content") or []:
            text = block.get("text") if isinstance(block, dict) else None
            if not text:
                continue
            for pattern in _TEXT_GUESS_PATTERNS:
                for m in pattern.finditer(text):
                    word = m.group(1).upper()
                    if word not in found:
                        found.append(word)
    return found


def parse_task(raw_prompt: str) -> tuple[str, str]:
    """Unpack the task from the dataset's prompt column (the RFT service
    passes only that column, as a string, to the agent)."""
    task = json.loads(raw_prompt)
    prompt = task.get("prompt") or "Guess the 5-letter word"
    answer = str(task.get("answer", "")).strip().upper()
    if len(answer) != WORD_LENGTH or not answer.isalpha():
        raise ValueError(f"Task must include a {WORD_LENGTH}-letter 'answer'; got {answer!r}")
    return prompt, answer


app = BedrockAgentCoreApp()


@app.entrypoint
@sagemaker_rft_handler
def handle_rollout(payload):
    metadata = payload.get("metadata", {})
    prompt, answer = parse_task(payload.get("prompt", ""))

    endpoint = metadata.get("endpoint") or os.environ.get("RFT_RUNTIME_ENDPOINT", "")
    if not endpoint:
        # Fall back to the regional Job Runtime endpoint instead of letting
        # the OpenAI client fail opaquely on a bare "/v1" base_url.
        endpoint = f"https://job-runtime.sagemaker.{AWS_REGION}.api.aws"
    endpoint = endpoint.rstrip("/")

    def build_model():
        """Policy model (the one being trained) served by the RFT Runtime.

        Reads the token fresh each call so a retry after an auth failure picks
        up a re-signed one.
        """
        params = {}
        if _reasoning_effort_ok:
            params["reasoning_effort"] = REASONING_EFFORT
        m = OpenAIModel(
            model_id="default",
            client_args={"api_key": get_cached_token(), "base_url": endpoint + "/v1"},
            **({"params": params} if params else {}),
        )
        # wrap_model injects RFT tracking headers and the service's inference
        # params (temperature/maxTokens/topP), merging with ours.
        return wrap_model(m)

    game = WordleGame(answer)

    @tool
    def guess_word(guess: str) -> str:
        """Submit a 5-letter Wordle guess and get feedback plus a Current
        Knowledge summary.

        Args:
            guess: A valid 5-letter English word.

        Returns:
            Per-letter feedback (green/yellow/gray) and the accumulated clue
            state, or an error message for invalid input.
        """
        return game.guess(guess)

    # A rollout that never plays a guess is fatal to the whole training job
    # ("No sampling requests were received for this rollout"), and signalling
    # {"status": "error"} is equally fatal ("The agent signaled that the
    # trajectory failed"). GPT-OSS-20B sometimes burns the entire token budget
    # reasoning before its first tool call, so retry with a fresh conversation
    # until at least one guess lands.
    global _reasoning_effort_ok
    agent = None
    deadline = time.monotonic() + ROLLOUT_BUDGET_SECONDS
    for attempt in range(AGENT_MAX_ATTEMPTS):
        if attempt and time.monotonic() >= deadline:
            logger.warning("rollout budget spent; reporting %d guess(es) "
                           "without further retries", len(game.guesses))
            break
        agent = Agent(model=build_model(), system_prompt=SYSTEM_PROMPT,
                      tools=[guess_word])
        try:
            agent(prompt if attempt == 0 else RETRY_PROMPT)
            break
        except Exception as e:
            # Token-limit truncation, throttling, or a stream fault. Guesses
            # already played are real signal, so never discard them.
            logger.warning("agent loop ended early on attempt %d/%d (%s: %s); "
                           "%d guess(es) played",
                           attempt + 1, AGENT_MAX_ATTEMPTS, type(e).__name__, e,
                           len(game.guesses))
            if _looks_like_auth_failure(e):
                # The cached bearer token was signed with container credentials
                # that have since rotated. Re-sign and retry, otherwise all
                # remaining attempts reuse the same dead token and 403 again.
                logger.warning("auth failure; refreshing bearer token")
                invalidate_token()
                continue
            if _reasoning_effort_ok and _looks_like_param_rejection(e):
                # The endpoint doesn't accept reasoning_effort. Stop sending it
                # process-wide and retry on the system-prompt hint alone.
                logger.warning("endpoint rejected reasoning_effort; disabling it")
                _reasoning_effort_ok = False
                continue
            if game.guesses or game.over:
                break

    # If the policy wrote its tool calls as text, no guess was executed. Replay
    # anything recoverable so the rollout still produces a graded game.
    if agent is not None and not game.over:
        for word in extract_text_guesses(getattr(agent, "messages", None)):
            if game.over:
                break
            if word in game.guesses:
                continue
            game.guess_from_text(word)
        if game.fallback_guesses:
            logger.warning("recovered %d guess(es) from raw-text tool calls",
                           game.fallback_guesses)

    # Always report a reward, never an error status: an error signal fails the
    # entire job, whereas a reward the service can't apply (already-failed
    # trajectory) is logged non-fatally and the rollout is retried.
    return {"reward": game.final_reward()}


if __name__ == "__main__":
    app.run()
