#!/usr/bin/env python3
"""Baseline a frontier Bedrock model on the Wordle validation set.

Runs the SAME environment, system prompt, tool and reward function that the
MTRL agent uses (imported from Wordle/app/Wordle/main.py), but points the
Strands agent at Bedrock instead of the RFT Runtime. That makes the numbers
directly comparable to MLflow's val/reward/pass_at_1 from a training job.

Usage:
    python eval_frontier.py --model global.anthropic.claude-opus-5 --n 25
    python eval_frontier.py --model global.anthropic.claude-sonnet-4-5-20250929-v1:0 --n 100

Notes:
  * pass@1 here uses the same success threshold the RFT service uses (reward
    >= 1.0), which for this reward function means "solved".
  * Runs games concurrently; keep --concurrency modest to avoid Bedrock
    throttling.
"""
import argparse
import json
import os
import statistics
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "Wordle" / "app" / "Wordle"
sys.path.insert(0, str(APP))

# The agent module reads word lists relative to its own file, so importing is safe.
os.environ.setdefault("REASONING_EFFORT", "none")  # Bedrock rejects reasoning_effort
import main  # noqa: E402
from main import WordleGame, parse_task, SYSTEM_PROMPT, MAX_GUESSES  # noqa: E402

from strands import Agent, tool  # noqa: E402
from strands.models import BedrockModel  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402

SUCCESS_THRESHOLD = 1.0  # matches val/reward/success_threshold in MLflow
_print_lock = threading.Lock()


def play_one(model_id, region, task_json, temperature, verbose=False, mantle=False):
    """Play a single game; returns (answer, solved, turns, reward, error)."""
    prompt, answer = parse_task(task_json)
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

    def build(temp):
        if mantle == "messages":
            # Bedrock Mantle, Anthropic Messages API (/anthropic/v1/messages).
            # Claude models on Mantle expose Messages + Responses, NOT Chat
            # Completions. Bearer token minted from the AWS credential chain.
            from aws_bedrock_token_generator import provide_token
            from strands.models.anthropic import AnthropicModel
            params = {} if temp is None else {"temperature": temp}
            return AnthropicModel(
                model_id=model_id, max_tokens=4096,
                client_args={"base_url": f"https://bedrock-mantle.{region}.api.aws/anthropic",
                             "auth_token": provide_token(region=region), "api_key": None},
                **({"params": params} if params else {}))
        if mantle:
            # Bedrock Mantle, OpenAI-compatible Chat Completions endpoint. Some
            # models (e.g. Gemma 3) only support client-side tool calling here,
            # not via the Converse API. Strands mints a short-lived bearer token
            # from the AWS credential chain; no API key needed.
            params = {} if temp is None else {"temperature": temp}
            return OpenAIModel(model_id=model_id,
                               bedrock_mantle_config={"region": region},
                               **({"params": params} if params else {}))
        kwargs = {"model_id": model_id, "region_name": region}
        if temp is not None:
            kwargs["temperature"] = temp
        return BedrockModel(**kwargs)

    err = None
    agent = None
    for temp in (temperature, None):
        try:
            agent = Agent(model=build(temp), system_prompt=SYSTEM_PROMPT,
                          tools=[guess_word])
            agent(prompt)
            err = None
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            # Newer reasoning models (e.g. Opus 5) reject `temperature`; retry
            # once without it rather than scoring a config error as a loss.
            if temp is not None and "temperature" in str(e).lower():
                game = WordleGame(answer)  # reset: nothing was played
                continue
            break

    # Same raw-text recovery the trained agent uses, for a fair comparison.
    if not game.over:
        try:
            for w in main.extract_text_guesses(getattr(agent, "messages", None)):
                if game.over or w in game.guesses:
                    continue
                game.guess_from_text(w)
        except Exception:
            pass

    reward = game.final_reward()
    if verbose:
        with _print_lock:
            mark = "OK " if game.solved else "-- "
            print(f"  {mark} {answer}  turns={len(game.guesses)}  reward={reward:+.3f}"
                  f"  {'| ' + err if err else ''}")
    return answer, game.solved, len(game.guesses), reward, err


def main_cli():
    ap = argparse.ArgumentParser(description="Frontier-model baseline on the Wordle val set")
    ap.add_argument("--model", default="global.anthropic.claude-opus-5")
    ap.add_argument("--dataset", default=str(ROOT / "validation-data.jsonl"))
    ap.add_argument("--n", type=int, default=25, help="games to play (0 = all)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 matches how the RFT service scores validation")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--mantle", nargs="?", const="chat", default=None,
                    choices=["chat", "messages"],
                    help="Call the model via Bedrock Mantle instead of the Converse API. "
                         "'chat' (default when flag given) = OpenAI Chat Completions, for "
                         "e.g. google.gemma-3-27b-it. 'messages' = Anthropic Messages API, "
                         "for Claude models (which lack Chat Completions on Mantle).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tasks = [json.loads(l)["prompt"] for l in open(args.dataset)]
    if args.n:
        tasks = tasks[: args.n]

    via = {None: "bedrock-runtime (Converse)", "chat": "bedrock-mantle (Chat Completions)",
           "messages": "bedrock-mantle (Anthropic Messages)"}[args.mantle]
    print(f"model      : {args.model}  via {via}")
    print(f"dataset    : {Path(args.dataset).name}  ({len(tasks)} games)")
    print(f"temperature: {args.temperature}   concurrency: {args.concurrency}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(play_one, args.model, args.region, t, args.temperature,
                          args.verbose, args.mantle) for t in tasks]
        for f in as_completed(futs):
            results.append(f.result())

    rewards = [r[3] for r in results]
    solved = [r for r in results if r[1]]
    turns = [r[2] for r in solved]
    errs = [r[4] for r in results if r[4]]
    n = len(results)

    print(f"\n{'='*58}\nRESULTS  ({n} games)")
    print(f"  pass@1 (reward >= {SUCCESS_THRESHOLD}) : {len(solved)}/{n} = {100*len(solved)/n:.1f}%")
    print(f"  mean reward                : {statistics.mean(rewards):+.4f}")
    print(f"  median reward              : {statistics.median(rewards):+.4f}")
    if n > 1:
        print(f"  stdev                      : {statistics.stdev(rewards):.4f}")
    if turns:
        print(f"  mean turns (solved only)   : {statistics.mean(turns):.2f}")
        print(f"  turn distribution          : {dict(sorted(Counter(turns).items()))}")
    if errs:
        print(f"  errors                     : {len(errs)}  e.g. {errs[0][:110]}")
    print(f"\n  reference: optimal play = +1.354 mean / 100% pass@1 / 3.42 turns")
    print(f"             trained GPT-OSS  = +0.111 mean /  34% pass@1 / 5.4 turns")


if __name__ == "__main__":
    main_cli()
