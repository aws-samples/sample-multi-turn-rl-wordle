# Wordle-MTRL: Training a Language Model to Play Wordle with Multi-Turn Reinforcement Learning on Amazon SageMaker AI

[![License: MIT-0](https://img.shields.io/badge/License-MIT--0-yellow.svg)](LICENSE)

This project trains an open-weight language model (**gpt-oss-20b**) to play
Wordle using **SageMaker AI Multi-Turn Reinforcement Learning (MTRL)**. An
agent hosted on Amazon Bedrock AgentCore plays the game, the policy model
being trained makes the guesses, and the outcome of each game is the reward.
Training updates a LoRA adapter with GRPO. After 100 training steps, the model
solves **84%** of held-out words at its default reasoning effort, up from 61%
before training, and **69%** at the low effort it was trained at, up from 15%.

Everything here has been run end to end, and every number below comes from a
real training or evaluation run. The goals are to show the complete MTRL
pipeline on a task everyone already understands, and to document the pitfalls
we hit so you can skip them.

This project is intended for demonstration purposes only. It is not intended
for use in a production environment.

## Table of Contents

- [Why Wordle? The Multi-Turn RL Challenge](#why-wordle-the-multi-turn-rl-challenge)
- [The Technology Stack: Why SageMaker AI and AgentCore?](#the-technology-stack-why-sagemaker-ai-and-agentcore)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [1. Test the environment locally](#1-test-the-environment-locally)
  - [2. Build the dataset](#2-build-the-dataset)
  - [3. Deploy the agent](#3-deploy-the-agent)
  - [4. Launch training](#4-launch-training)
  - [5. Evaluate the trained model at serving settings](#5-evaluate-the-trained-model-at-serving-settings)
  - [6. Baseline any Bedrock model](#6-baseline-any-bedrock-model)
  - [7. Clean up](#7-clean-up)
- [The Reinforcement Learning Strategy](#the-reinforcement-learning-strategy)
- [Results and Analysis](#results-and-analysis)
- [Lessons Learned](#lessons-learned)
- [Further Reading](#further-reading)
- [Attribution](#attribution)
- [License](#license)

---

## Why Wordle? The Multi-Turn RL Challenge

### What is Wordle?

Wordle is a word game with a hidden five-letter word and six guesses. After
each guess, every letter is marked green (right letter, right position),
yellow (in the word, wrong position), or gray (not in the word). For the full
rules, see [Wordle on Wikipedia](https://en.wikipedia.org/wiki/Wordle).

### Why multi-turn RL?

Wordle is a deliberately small task. Everyone already knows the rules, so you
can watch a model learn a multi-turn strategy without first learning a domain.
It is also genuinely multi-turn: each guess is constrained by the feedback from
the previous ones, and whether the game is won depends on decisions made
several turns earlier.

That is exactly the shape of problem MTRL is for. Supervised fine-tuning needs
a correct response to imitate, and a Wordle game has no single correct
sequence of guesses. Single-turn RL scores one response at a time, so it
cannot credit the opening guess that made a turn-3 solve possible. MTRL lets
the model play the whole game, scores the result, and learns from the spread
of results across several attempts at the same word.

Wordle is also a teaching example, not a business case. It can be solved
algorithmically: an information-theoretic solver
([3Blue1Brown](https://www.youtube.com/watch?v=v68zYyaEmEA)) averages about
3.4 guesses, and frontier models already play it well. In practice, you would
use MTRL to make a small, low-cost model good at a multi-step task specific to
your business, such as an internal workflow that public models were never
trained on. The same pipeline applies to that setting.

### An example rollout

The model plays through a single tool, `guess_word`. After each guess the
environment returns per-letter feedback and a plain-English summary of
everything known so far. Below is one complete rollout of the **trained
model**, taken from the AgentCore runtime traces of an evaluation job (medium
reasoning effort, temperature 1.0). The secret word was PASTE. The model's
reasoning is quoted as generated, trimmed where marked `[...]`.

```
[model] Reasoning: [...] The instruction says open with word that maximizes
        information (many common, distinct letters). "CRANE" has 5 distinct
        letters: C,R,A,N,E. [...] We'll guess "CRANE".
        Tool call: guess_word(guess="CRANE")

[tool]  Guess 1/6: CRANE -> C=gray, R=gray, A=yellow, N=gray, E=green
        Current Knowledge:
        - Correct Position (Green): _ _ _ _ E
        - In Word, Wrong Position (Yellow): 'A' (at least 1)
        - Not in Word (Gray): C, N, R
        - Words Already Guessed: CRANE
        - Guesses Remaining: 5

[model] Reasoning: We have word ends with E. A present somewhere else [...]
        Not C, N, R. [...] "ALIVE": ends with E, includes A. Letters not
        C,N,R. So guess "ALIVE".
        Tool call:
 guess_word(guess="ALIVE")

[tool]  Guess 2/6: ALIVE -> A=yellow, L=gray, I=gray, V=gray, E=green
        Current Knowledge:
        - Correct Position (Green): _ _ _ _ E
        - In Word, Wrong Position (Yellow): 'A' (at least 1)
        - Not in Word (Gray): C, I, L, N, R, V
        - Words Already Guessed: CRANE, ALIVE
        - Guesses Remaining: 4

[model] Reasoning: [...] in first guess A was at position 3: A=yellow. That
        means A in word but not position 3. [...] In second guess ALIVE, A
        was first letter, but yellow again: that means A in word but not
        position 1. So positions not 1 or 3 for A. Positions left: 2,4 for A.
        [...] We might guess "HASTE"? [...] contains A at pos
ition 2; good.
        Tool call: guess_word(guess="HASTE")

[tool]  Guess 3/6: HASTE -> H=gray, A=green, S=green, T=green, E=green
        Current Knowledge:
        - Correct Position (Green): _ A S T E
        - In Word, Wrong Position (Yellow): None
        - Not in Word (Gray): C, H, I, L, N, R, V
        - Words Already Guessed: CRANE, ALIVE, HASTE
        - Guesses Remaining: 3

[model] Reasoning: We have the word: _ASTE. [...] First letter unknown, not
        C H I L N R V. [...] "PASTE" is a word. PASTE fits. Also "WASTE" also
        fits [...] "BASTE" [...] Choose "PASTE" maybe.
        Tool call: guess_word(guess="PASTE")

[tool]  Guess 4/6: PASTE — all green. Correct, you solved it!
```

A few things to notice. The model opens with a strong word, keeps every green
letter in place, never reuses a gray letter, and uses each yellow to rule out
positions, as it does when it places the A on its third guess. None of this
was in the training data. The dataset contains only secret words, and the
model learned the strategy from the reward.

When the game ends, the agent returns one number, the reward: **+1.279** for
this game. How that number is built is covered in
[The Reinforcement Learning Strategy](#the-reinforcement-learning-strategy).

---

## The Technology Stack: Why SageMaker AI and AgentCore?

![Architecture: SageMaker MTRL training job invoking an AgentCore-hosted Wordle agent, which samples the policy model and reports a reward](docs/architecture.png)

The system has two halves that talk to each other during training.

**Amazon SageMaker AI Multi-Turn RL** owns the training job and the policy
model. It is serverless: you pick a base model, point the job at your agent
and dataset, and pay per token processed, with no GPU cluster to provision.
For every rollout the service invokes the agent with one training row. It
collects several rollouts of the same word, and **Group Relative Policy
Optimization (GRPO)** uses the spread of their rewards to update the model.
Metrics stream to an MLflow app.

**Amazon Bedrock AgentCore** hosts the agent. A single Python file,
`Wordle/app/Wordle/main.py`, holds the environment, the `guess_word` tool, and
the reward function, so there is no separate reward service. The agent is a
Strands agent that calls the policy model through the SageMaker Job Runtime's
OpenAI-compatible endpoint.

**LoRA (low-rank adaptation)** is how MTRL keeps training affordable. Rather
than updating all of the model's weights, LoRA freezes the base model and
trains two small matrices next to selected weight matrices; their product is a
low-rank update added to the frozen weights. In this run the service trained
rank-32 adapters (alpha 64) on gpt-oss-20b's attention projections and
mixture-of-experts layers. The adapter is 1.4 GB, against about 39 GB for the
full model, and the output model package contains both the adapter and a
merged checkpoint ready to deploy.

Supported base models at the time of writing (see the
[MTRL documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html)
for the current list):

| Base model | AWS Regions |
|---|---|
| Nova 2 Lite | us-east-1, us-west-2 |
| GPT-OSS-20B | us-east-1, us-west-2 |
| Gemma 4 31B (instruction-tuned) | us-west-2 |
| Qwen 3.6 27B | us-west-2 |

This project uses GPT-OSS-20B: it has the lowest per-token price of the four,
its reasoning effort is controllable per request, and the open weights make
the adapter easy to serve anywhere.

---

## Getting Started

### Prerequisites

- Python 3.10+, [`uv`](https://docs.astral.sh/uv/), Node 20+ (for the CDK),
  and the AgentCore CLI (`npm install -g @aws/agentcore`). See
  [Get started with Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html)
  for installation and the IAM permissions the CLI needs.
- `botocore >= 1.43.37` on the machine that launches the job (first version
  with the AgentRFT `CreateJob` operation).
- An S3 bucket and an MLflow app in the Region you deploy to.
- The two IAM roles described below.

Pinned versions this was validated with: `sagemaker-train 1.21.0`,
`sagemaker-core 2.21.0`, `bedrock-agentcore 1.22.0`, `strands-agents 1.54.0`.

### 1. Set up the environment

Clone the repository and install both Python environments: the root one for
the launcher and evaluation scripts, and the agent's own.

```bash
git clone https://github.com/aws-samples/sample-multi-turn-rl-wordle.git
cd sample-multi-turn-rl-wordle
uv sync
(cd Wordle/app/Wordle && uv sync)
```

### 2. Create the IAM roles

Two roles. Both trust policies matter, and one of them is not what the generic
SageMaker docs show.

**Agent runtime role** (created by `agentcore deploy`): trusts
`bedrock-agentcore.amazonaws.com`; needs `AmazonSageMakerJobRuntimeAccess`
(attached by the CDK stack in this repo).

**Training job role** (you create it): must trust
**`job.sagemaker.amazonaws.com`** with both `sts:AssumeRole` and
`sts:TagSession`. Plain `sagemaker.amazonaws.com` is not enough, and the error
message says so only after the third retry. Attach
`AmazonSageMakerJobFullAccess`, which covers S3, ECR, EC2 networking, MLflow,
model packages, and `bedrock-agentcore:InvokeAgentRuntime`.

Your caller needs `iam:PassRole` for `job.sagemaker.amazonaws.com` plus the
`sagemaker:*Job*` actions. Full reference:
[MTRL prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl-prereqs.html).

### Repository layout

```
.
├── Wordle/                         AgentCore project (agentcore-cli)
│   ├── agentcore/
│   │   ├── agentcore.json          runtime spec + env vars (REASONING_EFFORT, ...)
│   │   ├── aws-targets.json        account + region  <-- edit this
│   │   └── cdk/lib/cdk-stack.ts    attaches AmazonSageMakerJobRuntimeAccess
│   └── app/Wordle/
│       ├── main.py                 agent, environment, reward, RFT handler
│       ├── test_env.py             offline tests, no AWS needed
│       └── data/                   SCOWL word lists + entropy: reward reference data, not training data
├── docs/architecture.{html,png}    the diagram above (SVG source + render)
├── docs/plots/make_plots.py        regenerates the result charts from MLflow
├── build_word_lists.py             builds data/ from SCOWL + AGID (pinned, checksummed)
├── excluded_answers.txt            manual review: words never used as answers
├── licenses/                       verbatim SCOWL, AGID, and UKACD notices
├── make_dataset.py                 builds training/validation JSONL
├── run_mtrl_training.py            launches / attaches to the MTRL job
├── run_mtrl_eval.py                SageMaker evaluation jobs: base vs trained at chosen temperature
├── eval_frontier.py                zero-shot baseline for any Bedrock model
├── training-data.jsonl             640 unique secret words (5 batches of 128)
└── validation-data.jsonl           128 unique, disjoint from training
```

---

## Usage

### 1. Test the environment locally

No AWS account needed. This checks the game logic, the reward ordering, and
the dataset round-trip.

```bash
cd Wordle/app/Wordle
uv run python test_env.py
```

### 2. Build the dataset

The word lists in `Wordle/app/Wordle/data/` are checked in, so this step is
optional. To rebuild them from SCOWL and AGID (downloaded once, checksum
verified):

```bash
uv run python build_word_lists.py
```

Then draw the training and validation words:

```bash
uv run python make_dataset.py      # 640 train / 128 val, all unique, seed 42
```

This writes `training-data.jsonl` (640 rows) and `validation-data.jsonl` (128
rows, no overlap with training). Here are the first three training rows:

```jsonl
{"prompt": "{\"prompt\": \"Guess the 5-letter word\", \"answer\": \"riser\", \"id\": \"wordle_train_0000\"}"}
{"prompt": "{\"prompt\": \"Guess the 5-letter word\", \"answer\": \"swoon\", \"id\": \"wordle_train_0001\"}"}
{"prompt": "{\"prompt\": \"Guess the 5-letter word\", \"answer\": \"dirty\", \"id\": \"wordle_train_0002\"}"}
```

Each row has one column, `prompt`, and its value is a string. The MTRL
service reads that column and passes the string to the agent verbatim; it
does not parse or validate it. So the row packs everything the agent needs
into a small JSON object, which `parse_task()` in `main.py` unpacks:

```json
{
  "prompt": "Guess the 5-letter word",
  "answer": "riser",
  "id": "wordle_train_0000"
}
```

- `prompt` is the instruction shown to the policy model. It is identical on
  every row, because every game starts the same way.
- `answer` is the secret word. It stays inside the environment: the model
  never sees it, only the feedback the environment computes from it.
- `id` identifies the row in logs.

That is the whole dataset. There are no example games, no target responses,
and no labels in the usual sense. The service also accepts Parquet, JSON, and
CSV, and for tasks that need more context a row can carry a full message
list, tool configuration, or reward specification in the same string. See
[Prompt dataset format](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl-assets.html)
in the SageMaker AI documentation.

### 3. Deploy the agent

Edit `Wordle/agentcore/aws-targets.json`: replace the placeholder account ID
with yours, and pick the Region. The training job must run in the **same
Region** as the runtime, or `CreateJob` rejects it.

If you plan to contribute back, keep your account ID out of commits with
`git update-index --skip-worktree Wordle/agentcore/aws-targets.json` after
editing it. Then:

```bash
cd Wordle
agentcore validate
agentcore deploy
```

Note the runtime ARN from the output. The CDK stack attaches the
`AmazonSageMakerJobRuntimeAccess` managed policy to the runtime role; without
it every rollout fails with `AccessDenied` when the agent calls the policy
model. Use the CLI rather than hand-building a code package: the runtime runs
Linux ARM64, and the agent's compiled dependencies (numpy, pandas, pyarrow,
through MLflow) must be resolved for that platform.

Smoke-test with an RFT-shaped payload. [`payload.json`](Wordle/payload.json)
holds one task row in the `prompt` field and a fake `jobArn` in `metadata`.
Put your own account ID into that fake ARN first: the service rejects a job
ARN from another account with a 403 before it checks whether the job exists,
and a 403 is indistinguishable from a broken IAM setup. From `Wordle/`:

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
sed "s/111122223333/$ACCOUNT/" payload.json > /tmp/payload.json

aws bedrock-agentcore invoke-agent-runtime \
  --agent-runtime-arn <RUNTIME_ARN> \
  --runtime-session-id smoke-test-$(date +%s)000000000000000 \
  --payload fileb:///tmp/payload.json /dev/stdout
```

The command prints `{"reward": -1.5}`: the agent never reached the policy
model, so it reported the no-play reward. The proof is in the runtime's
CloudWatch logs: the agent parses the prompt, authenticates to the Job Runtime
endpoint, and gets a 400 reading "Could not find job". That confirms payload
parsing, bearer-token auth, and error reporting all work. A 403 "not
authorized to access this resource" means the job ARN still names another
account; a 30-second timeout means the HTTP server isn't starting.

### 4. Launch training

```bash
uv run python run_mtrl_training.py \
  --model openai-reasoning-gpt-oss-20b \
  --agent-runtime-arn <RUNTIME_ARN> \
  --role-arn arn:aws:iam::<ACCOUNT>:role/<JOB_ROLE> \
  --s3-prefix s3://<BUCKET>/wordle-mtrl \
  --s3-output-path s3://<BUCKET>/wordle-mtrl/output/ \
  --val-dataset s3://<BUCKET>/wordle-mtrl/validation/validation-data.jsonl \
  --mlflow-app-arn arn:aws:sagemaker:<REGION>:<ACCOUNT>:mlflow-app/<APP_ID>
```

`--s3-prefix` uploads `training-data.jsonl` for you; pass `--train-dataset`
instead to reuse an existing S3 object. `MlflowConfig` is **required** by
`CreateJob`, so create an MLflow app first
(`aws sagemaker create-mlflow-app ...`) if you don't have one. The 100-step
run in this repository took 12 hours 40 minutes and cost about $339 in
training tokens.

Reattach to a running job at any time:

```bash
uv run python run_mtrl_training.py --attach <JOB_NAME> --role-arn ... --s3-output-path ...
```

### 5. Evaluate the trained model at serving settings

The training job's validation curve is scored at temperature 0 and the
deployed reasoning effort. To score the model package (and the base model) at
the settings you will actually serve, launch SageMaker evaluation jobs against
the same runtime and reward:

```bash
uv run python run_mtrl_eval.py --runs base:1.0 trained:1.0 \
  --model-package-arn <OUTPUT_MODEL_PACKAGE_ARN> \
  --agent-runtime-arn <RUNTIME_ARN> \
  --role-arn arn:aws:iam::<ACCOUNT>:role/<JOB_ROLE> \
  --dataset s3://<BUCKET>/wordle-mtrl/validation/validation-data.jsonl \
  --s3-output-path s3://<BUCKET>/wordle-mtrl/eval/ \
  --mlflow-app-arn arn:aws:sagemaker:<REGION>:<ACCOUNT>:mlflow-app/<APP_ID>
```

Each run is 128 rollouts and takes four to seven minutes; metrics land in MLflow
under `eval/reward/*`. Only one evaluation job runs at a time per account, so
the script runs `--runs` sequentially. Reasoning effort is not a job
parameter: change `REASONING_EFFORT` in `agentcore.json`, `agentcore deploy`,
evaluate, then deploy the training value back. Never redeploy while a training
or evaluation job is running.

The `HyperParameters` the API accepts are flat strings named
`eval_group_size`, `temperature`, `sampling_top_p`, `sampling_max_tokens`,
`pass_k_values`, `success_threshold`, `rollout_timeout`,
`rollout_max_concurrency`, `rollout_max_retries`. The nested form in the docs
and the `sampling_temperature` / `top_p` / `max_tokens` names the SDK's
`MultiTurnRLEvaluator` emits are both rejected by schema validation; the error
message lists the accepted names.

### 6. Baseline any Bedrock model

```bash
uv run python eval_frontier.py --model global.anthropic.claude-opus-5 --n 100
```

Runs the same environment and reward against a Bedrock model. The harness
sends no reasoning or thinking parameters, so each model runs at its Bedrock
default: adaptive thinking at high effort for Claude Opus 5, thinking off for
Claude Haiku 4.5, medium effort for gpt-oss. Record that alongside the number.

By default this uses the Converse API. Some models only support tool calling
through Bedrock Mantle, and Claude models on Mantle expose the Anthropic
Messages API rather than Chat Completions, so pick the path per model:

```bash
uv run python eval_frontier.py --model google.gemma-3-27b-it --mantle            # Chat Completions
uv run python eval_frontier.py --model anthropic.claude-opus-5 --mantle messages  # Anthropic Messages
```

Check the model's Bedrock model card for which endpoint and API carry
client-side tool calling.

### 7. Clean up

Training and evaluation jobs stop billing when they complete, but the
AgentCore runtime, its IAM role, the MLflow app, and the S3 objects persist.

```bash
# 1. Agent runtime and its execution role. `remove all` resets the project
#    config; the next deploy sees the empty state and tears down the resources.
cd Wordle
agentcore remove all
agentcore deploy

# 2. MLflow app
aws sagemaker delete-mlflow-app --arn arn:aws:sagemaker:<REGION>:<ACCOUNT>:mlflow-app/<APP_ID>

# 3. Datasets, job output, and evaluation output
aws s3 rm s3://<BUCKET>/wordle-mtrl/ --recursive
```

`remove all` rewrites the files under `Wordle/agentcore/`, so run
`git checkout -- Wordle/agentcore/` afterward if you want to redeploy later.
Deleting the CloudFormation stack directly
(`aws cloudformation delete-stack --stack-name AgentCore-Wordle-default`)
removes the same resources without touching the project files. The training
job role and the model packages in the output model package group are not
billed, so you can keep them for a future run.

---

## The Reinforcement Learning Strategy

The reward function is where multi-turn RL succeeds or fails. Each rollout
returns a single number built from two parts: a dominant **outcome** term that
depends only on how the game ended, and a small bounded **shaping** term that
rewards good Wordle play within each outcome.

```
reward = outcome + w * clip(shaping, -1, 1)    # w = 0.04 if solved, 0.15 otherwise
```

### 1. The outcome

| Outcome | Value |
|---|---:|
| Solved on guess *n* | `1.05 + 0.5 * (6-n)/5` → 1.55 on guess 1, 1.05 on guess 6 |
| Played all six, unsolved | −0.5 |
| Never made a valid guess | −1.5 |

A faster solve always beats a slower one, every solve beats every loss, and
never playing is strictly the worst result. Adjacent guess counts are 0.1
apart, and shaping can move a solve by at most ±0.04, so two solves can never
swap order. Every solve also scores at least 1.01, so it clears the service's
`success_threshold` of 1.0 and pass@1 counts it. Losses keep a larger shaping
weight (±0.15) because they have no guess-count tiers to overlap, and the best
possible loss (−0.35) is still far below the worst solve. `test_env.py` checks
every tier boundary at both shaping extremes, so an edit cannot silently break
the ordering.

### 2. Shaping: penalties for rule violations and mistakes (the "stick")

The shaping score is a running total over the game, using the per-guess
scoring from [wordle-lora-rl](https://github.com/charbull/wordle-lora-rl). The
environment tracks every known green, yellow, and gray letter and penalizes
guesses that ignore them:

- **Green violation (`green_position_penalty`, −30):** not placing a known green letter in its spot.
- **Yellow violation (`yellow_letter_penalty`, −20):** leaving out a known yellow letter.
- **Gray violation (`gray_letter_penalty`, −20):** using a letter already marked gray.
- **Repeated guess (`repetition_penalty`, −40):** guessing a word already played. It still uses up a guess.
- **Not a word (`not_in_dictionary_penalty`, −35):** five letters, but not in the Wordle dictionary.
- **Malformed input (−50):** not five letters. It does not use up a guess.
- **Raw-text tool call (`text_format_penalty`, −5):** a guess recovered from a tool call the model wrote as plain text instead of a structured call (see [Lesson 3](#lesson-3-rl-can-erode-tool-calling-format)).

### 3. Shaping: bonuses for strategic play (the "carrot")

- **Valid guess (`valid_guess_base`, +15):** for each valid guess that doesn't solve the game.
- **Opening word (`information_gain_bonus_coeff`, 7.5 × entropy):** on turn 1 only, a bonus proportional to the word's pre-computed information gain, so strong openers like SLATE or TRACE score well.
- **New letters (`new_letter_bonus`, +2 each):** from turn 2 on, for each letter not yet tried.
- **Possibility reduction (`possibility_reduction_bonus`, up to +15):** from turn 2 on, proportional to the fraction of remaining candidate answers the guess eliminates.

### 4. Penalties for inefficiency

- **Time (`time_penalty_per_guess`, −1):** every guess.
- **Stagnation (`green_reuse_penalty` −3, `yellow_reuse_penalty` −1.5):** for reusing already-known letters instead of testing new ones.

### 5. How the pieces combine

The shaping total is divided by 150, clipped to ±1, and weighted by 0.04 for
a solve or 0.15 for a loss. That is enough to separate two games with the same
outcome, but never enough to cross into another tier.

Take the PASTE rollout above. Solving on guess 4 gives an outcome of 1.25, and
the shaping total of +108.1 (a strong opener, new letters on every turn, no
violations) adds 0.029, for **+1.279**. Had the model wasted a guess on CASTE
before PASTE, which reuses the gray C, the same word would have scored
**+1.174**: the outcome drops to 1.15 for a five-guess solve, and the
violation penalty lowers the shaping.
Both games won, but one was played better, and GRPO needs exactly that kind
of difference between rollouts to compute a gradient. The constants live at
the top of `Wordle/app/Wordle/main.py`.

### Hyperparameters

Set in `run_mtrl_training.py`. Every training hyperparameter is the
gpt-oss-20b default except these three:

| Parameter | Value | Default | Why |
|---|---|---|---|
| `sampling_max_tokens` | 8192 | 4096 | The service cap. At 4096, gpt-oss often ran out of tokens while reasoning, before its first tool call. |
| `max_epochs` | 20 | 1 | 640 prompts at the default batch of 128 is 5 steps per epoch, so 20 epochs reach 100 steps. |
| `max_steps` | 100 | 100 | Same as the default; set explicitly so the two limits agree. |

The defaults that matter most here are `global_batch_size` 128 and
`group_size` 8 (1,024 rollouts per step), `learning_rate` 1e-5,
`temperature` 1.0, and `rollout_max_concurrency` 96. Run
`trainer.hyperparameters.get_info()` for the full list. The SDK sends only
values that differ from its defaults, so `DescribeJob` lists just
`sampling_max_tokens` and `max_epochs` under `HyperParameters`.

`REASONING_EFFORT=low` is set as a runtime env var in `agentcore.json`, and
gpt-oss honors it as a request parameter. `REASONING_PROMPT_HINT`
additionally prepends `Reasoning: low` to the system prompt, which is the
Harmony convention gpt-oss expects; leave it off for models that don't use
the Harmony format. In our evaluation jobs at temperature 1.0, low effort used
63 to 83% fewer sample tokens per game than medium. The adapter it produced
transfers to medium effort at serving time (69% → 84%, see
[Results and Analysis](#results-and-analysis)), so training at low and serving
at the model's default is a reasonable trade; training at medium is untested
here and would cost roughly 2.7 times the sample tokens (3,797 vs 1,412 per
game for the trained model in the evaluation jobs).

---

## Results and Analysis

Training ran for 100 steps on 640 words and was scored on 128 held-out words
the model never saw in training.

### Training performance

The training service logs metrics to MLflow as it trains. Two tell the story:

![Left: mean reward per game rising from -0.23 to about +0.57 for training rollouts within 15 steps, and from -0.40 to between +0.08 and +0.33 for validation. Right: validation solve rate rising from 2% at step 0 to 31% at step 10, then between 35% and 48% for the rest of the run, ending at 37%.](docs/plots/training_curves.png)

- **Mean reward per game (left).** Training rollouts (`rollout/reward/mean`)
  climb from −0.23 to about +0.57 within 15 steps and hold there for the rest
  of the run. Validation reward on held-out words (`val/reward/mean`) rises
  from −0.40 to between +0.08 and +0.33.
- **Validation solve rate (right).** The share of held-out words solved
  (`val/reward/pass_at_1`) goes from 2% to 31% in the first 10 steps, then
  moves between 35% and 48% and ends at 37%. With 128 words, one point is
  about 1.3 words, so the swings after step 10 are mostly noise.

The two sets of curves are sampled differently. Training rollouts use
temperature 1.0; the service scores validation at **temperature 0**, with the
effort the agent was deployed with (`low` here), and neither setting is
configurable. The validation curve is a faithful signal that training works
and shows where it levels off, but it understates the adapter by a wide
margin, as the next section shows.

### Evaluation: trained vs. base model

We evaluated the base model and the trained model package with SageMaker
evaluation jobs, varying the two settings the curve holds fixed: **reasoning
effort** and **sampling temperature**. In each bar, the gray segment is the
base model and the blue segment is the gain from training, so the top of the
bar is the trained model.

![Stacked bars for four settings. Solve rate, base to trained: low effort temp 0, 2% to 37%; medium effort temp 0, 16% to 30%; low effort temp 1.0, 15% to 69%; medium effort temp 1.0, 61% to 84%. Mean reward, base to trained: -0.40 to +0.14; -0.15 to +0.11; -0.19 to +0.69; +0.60 to +0.99.](docs/plots/eval_trained_vs_base.png)

The low-effort, temperature-0 pair is the training job's own validation at
steps 0 and 100. The trained model at medium effort and temperature 1.0 is the
average of two evaluation runs (80% and 88%); the base model at that setting
ran once. All seven evaluation jobs together cost about $1.20 in tokens.

### Analysis and key findings

1. **Training lifts the solve rate in every setting.** At the model's default
   reasoning effort (medium) and temperature 1.0, it goes from 61% to 84%. At
   the low effort it was trained at, it goes from 15% to 69%.
2. **The base model is already decent at serving settings.** Medium effort
   lets the base model reason its way to 61%, so there is less headroom there
   than at low effort, where training adds 54 points.
3. **Greedy decoding hides the gains.** At temperature 0 the trained model
   solves 30 to 37% of words at either effort, against 69 to 84% at
   temperature 1.0. Greedy decoding often ends the game without a solve.
4. **The adapter transfers across settings.** It adds 14 to 54 points in
   every cell, including medium effort, which it was never trained at.
5. **Evaluate at the settings you will serve at.** Temperature 1.0 is also what
   training sampled at, so the trained model is in-distribution there. Had we
   trusted only the training curve, we would have reported 37%.

**Next steps to improve performance:** training reward stopped rising after
about step 15, and validation stopped improving after about step 30, so more
steps alone won't help. The most promising untried change is a curriculum that
seeds games with 0–4 prior guesses, so the policy learns deduction without
first surviving the opening; wordle-lora-rl reports this as its single
biggest gain. Raising `group_size` above the default of 8 is the other obvious
lever. Every training row has the identical prompt text and only the hidden
answer varies, so this dataset exercises "learn from interaction," not "learn
from your data."

---

## Lessons Learned

Things that cost real training runs to discover, in the order you'll hit them.

### Lesson 1: The plumbing fails before the learning does

Most of the early failures were integration bugs, not RL problems.

- **The `@sagemaker_rft_handler` decorator does not start an HTTP server.**
  The AWS docs template implies it does. Without `BedrockAgentCoreApp` and
  `@app.entrypoint` wrapping it, the runtime times out after 30 s on every
  invocation. Decorator order is `@app.entrypoint` outermost.
- **`generate_token()` builds a fresh botocore session per call.** Under 32
  concurrent rollouts that starves the container credential provider
  (`No AWS credentials found`). Caching the token for an hour then failed
  differently: it's signed with rotating instance credentials, so cached
  tokens go stale and 403 (`Authentication failed`). The fix is one shared
  botocore session whose credentials self-refresh, plus a short token cache
  and a re-sign on 403.
- **Never redeploy the agent while a job is running.** It restarts the
  runtime, in-flight rollouts see no sampling requests, and the service fails
  the whole job.
- **Always return a reward.** A rollout that never reaches the policy model
  fails the entire job (`No sampling requests were received`); we hit this when
  every model call failed authentication. Returning `{"status": "error"}` fails
  it too (`The agent signaled that the trajectory failed`). A game where the
  model is sampled but never makes a valid guess is fine: it scores −1.5, and
  the 100-step run had 51 among about 105,000 rollouts. The agent still retries a zero-guess rollout once
  with a fresh conversation, bounded by a wall-clock budget so it can't outrun
  the reward-reporting window.

### Lesson 2: Summing per-guess rewards inverts the incentives

The first design summed wordle-lora-rl's per-guess scores into one reward per
game. That works in wordle-lora-rl, which scores each guess as an independent
training example, but over a whole game it paid +0.49 for a *lost* game and
scored a turn-1 solve *below* a turn-3 solve, because six turns of base and
exploration bonuses outweighed a −1/turn time penalty. Separating a dominant
outcome term from bounded shaping fixed both. If you change the reward, run
`test_env.py`; it asserts `fast win > slow win > loss > never played`.

### Lesson 3: RL can erode tool-calling format

RL can gradually push a model away from the structured tool calls its agent
framework knows how to execute. Instead of a real `guess_word` call, the model
starts writing the call out as plain text, for example
`<__function=guess_word>...` or `guess_word("crane")`. Once that happens,
nothing runs: every rollout in a group gets the same floor reward,
within-group variance drops to zero, and GRPO has no gradient left to pull the
model back. The drift can take over a run and never reverse.

It is more likely at higher learning rates, and some models are more prone to
it than others, so treat it as a risk to watch for rather than a rare bug.
Three defenses:

- **Start at a conservative learning rate.** This run used 1e-5. Raise it
  only with evidence that training is too slow.
- **Watch for it early.** Track how often guesses arrive as raw text rather
  than structured calls; the agent logs a warning for every rollout in which
  it had to recover them.
- **Keep a gradient alive.** `main.py` recovers guesses written as raw text
  (three observed formats) and scores them with a small formatting penalty,
  so rollouts still differ and GRPO can steer back toward proper calls. The
  same recovery turned out to matter at evaluation time, for models that write
  tool calls as prose.

### Lesson 4: Measure with the service's metrics, not log scraping

The service logs `val/reward/pass_at_1` on the held-out set with
`success_threshold = 1.0`, which for this reward function means "solved."
Scraping CloudWatch for "solved" strings over-reported the win rate by more
than 2x because it mixed training and validation rollouts and dropped rollouts
that ended in exceptions.

### Lesson 5: The validation curve is not the serving number

The training job scores validation at temperature 0 with the deployed
reasoning effort, and you cannot change either. For this task greedy decoding
makes the model quit early, so the curve ended at 37% while the same model
package scores 84% at medium effort and temperature 1.0. Run
`run_mtrl_eval.py` at your serving settings before drawing conclusions, and use
the curve for what it is good at: showing whether training is still improving.

### Lesson 6: Training settings and serving settings are separate choices

Reasoning effort and temperature each show up twice in this pipeline: once
when training samples rollouts, and again when you evaluate or serve the
model. They don't have to match, and choosing them separately paid off here.

- **Train cheap, serve at the default.** In our evaluation jobs, low reasoning
  effort used 63 to 83% fewer sample tokens per game than medium. The
  resulting adapter scored 84% when served at medium effort, against 69% at
  low, and its gain over the base model held at both (+23 and +54 points).
- **Serve near the training temperature.** Training sampled at temperature
  1.0. At 1.0 the trained model solved 69 to 84% of words; at temperature 0 it
  fell to 30 to 37%, at either effort.
- **Know what you haven't tested.** This project changed these settings at
  evaluation time only. Whether training at medium effort or a different
  temperature would raise the ceiling is untested, and medium effort would
  cost roughly 2.7 times the sample tokens.

---

## Further Reading

- [Multi-turn reinforcement learning on Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html): concepts, supported models, and pricing dimensions
- [MTRL model evaluation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl-evaluation.html): evaluation jobs and their metrics
- [MTRL prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl-prereqs.html): IAM roles and permissions
- [Amazon SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/): per-token rates for prefill, sample, and train
- [Get started with Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html): the AgentCore CLI
- [wordle-lora-rl](https://github.com/charbull/wordle-lora-rl): the Wordle environment and reward this project builds on, trained locally with MLX
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024), which introduced GRPO
- [Solving Wordle using information theory](https://www.youtube.com/watch?v=v68zYyaEmEA) (3Blue1Brown)

## Attribution

The environment design, reward constants, and clue-state tracking are ported
from [charbull/wordle-lora-rl](https://github.com/charbull/wordle-lora-rl)
(its README declares the MIT license), adapted from tag-parsing GRPO to a
tool-calling agent on AgentCore. The word lists are built from
[SCOWL](http://wordlist.aspell.net/) by `build_word_lists.py`.
 The optimal-play statistics (3.42 mean
guesses) follow the information-theoretic approach popularized by
[3Blue1Brown](https://www.youtube.com/watch?v=v68zYyaEmEA).

## License

MIT-0 (MIT No Attribution). See [LICENSE](LICENSE).

The word lists are built from [SCOWL](http://wordlist.aspell.net/) and
filtered with AGID, both by Kevin Atkinson under permissive licenses. Portions
of the Wordle environment and reward come from
[wordle-lora-rl](https://github.com/charbull/wordle-lora-rl) under the MIT
License. See [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES) and `licenses/`.
