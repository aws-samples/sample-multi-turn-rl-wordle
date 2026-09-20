# Teaching a 20B model to play Wordle with SageMaker Multi-Turn RL

A complete, end-to-end example of **SageMaker AI Multi-Turn Reinforcement
Learning (MTRL)**: an agent hosted on Amazon Bedrock AgentCore plays Wordle,
the policy model being trained makes the guesses, and the game's outcome is
the reward. Everything here has been run to completion, and the numbers below
are real.

Wordle is a deliberately small task. That's the point: everyone already knows
the rules, so you can watch a model learn a multi-turn strategy without first
learning a domain. The same pipeline runs unchanged on harder environments.

## Results

Base `openai-reasoning-gpt-oss-20b` versus the same model after 100 MTRL
steps, scored on 100 held-out secret words the policy never saw in training.
Frontier models are run zero-shot through the **identical** environment,
prompt, tool, and reward for comparison.

> GitHub: `https://github.com/aws-samples/sample-multi-turn-rl-wordle`

| Model | Solve rate | Mean reward | Mean turns | Trained on task? |
|---|---:|---:|---:|:---:|
| Optimal solver (information-theoretic) | 100% | +1.354 | 3.42 | — |
| Claude Opus 5 | 95% | +1.226 | 3.82 | no |
| Claude Haiku 4.5 | 82% | +0.950 | 4.44 | no |
| DeepSeek v3.2 | 58% | +0.537 | 4.34 | no |
| **gpt-oss-20b, after MTRL** | **34%** | **+0.111** | 5.4 | **yes** |
| Nova 2 Pro (preview) | 30% | −0.001 | 4.70 | no |
| Nova 2 Lite | 17% | −0.231 | 4.53 | no |
| Qwen3-32B | 15% | −0.282 | 4.67 | no |
| **gpt-oss-20b, base** | **7%** | −0.316 | — | no |

Validation solve rate over training (`val/reward/pass_at_1` from MLflow):

```
step:   0     10    20    30    40    50    60    70    80    90    100
       0.07  0.25  0.29  0.28  0.31  0.38  0.39  0.35  0.33  0.33  0.34
```

Read this honestly: RL took a model that could barely play (7%) to one that
beats several larger untrained models, a ~5x improvement, and the gain
plateaued around step 60. It is still far from a frontier model. Both facts
are part of the story. See [What we learned](#what-we-learned) for what
would move the number further.

## How it works

![Architecture: SageMaker MTRL training job invoking an AgentCore-hosted Wordle agent, which samples the policy model and reports a reward](docs/architecture.png)

Each training row is one secret word. For every rollout the MTRL service
invokes the AgentCore runtime with that row; the agent calls the policy model
through the SageMaker Job Runtime's OpenAI-compatible endpoint, the model
plays up to six guesses through a `guess_word` tool, and the agent returns a
single scalar reward. GRPO uses the spread of rewards across rollouts of the
same word to update the LoRA adapter.

### The reward

The rollout reward is a dominant **outcome** term plus a small bounded
**shaping** term:

```
reward = outcome + 0.15 * clip(shaping, -1, 1)
```

| Outcome | Value |
|---|---:|
| Solved on guess *n* | `1.0 + 0.5 * (6-n)/5` → 1.5 on guess 1, 1.0 on guess 6 |
| Played all six, unsolved | −0.5 |
| Never made a valid guess | −1.5 |

Shaping is the per-guess score from
[wordle-lora-rl](https://github.com/charbull/wordle-lora-rl): clue-consistency
penalties (contradicting a green/yellow/gray), repetition and non-dictionary
penalties, an entropy bonus for the opener, and new-letter and
possibility-reduction bonuses. It only moves a result *within* its tier, so a
loss can never outscore a win, but rollouts still differ enough for GRPO to
get a gradient. The exact constants are at the top of
`Wordle/app/Wordle/main.py`; the ordering is asserted in `test_env.py`.

## Repository layout

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
│       └── data/                   NYT word lists + per-word entropy
├── docs/architecture.{html,png}    the diagram above (SVG source + render)
├── make_dataset.py                 builds training/validation JSONL
├── run_mtrl_training.py            launches / attaches to the MTRL job
├── eval_frontier.py                zero-shot baseline for any Bedrock model
├── training-data.jsonl             600 unique secret words
└── validation-data.jsonl           100 unique, disjoint from training
```

## Prerequisites

- An AWS account **allowlisted for SageMaker AgentRFT**. Without it,
  `CreateJob` fails with a bare `AccessDenied` even from an Admin role.
- Python 3.10+, [`uv`](https://docs.astral.sh/uv/), Node 20+ (for the CDK),
  and the [`agentcore` CLI](https://docs.aws.amazon.com/bedrock-agentcore/).
- `botocore >= 1.43.37` on the machine that launches the job (first version
  with the AgentRFT `CreateJob` operation).
- Two IAM roles, described in [IAM](#iam).

Pinned versions this was validated with: `sagemaker-train 1.21.0`,
`sagemaker-core 2.21.0`, `bedrock-agentcore 1.22.0`, `strands-agents 1.54.0`.

## Quick start

### 1. Test the environment locally (no AWS)

```bash
cd Wordle/app/Wordle
uv sync
uv run python test_env.py          # environment, reward ordering, dataset round-trip
```

### 2. Build the dataset

```bash
uv run python make_dataset.py      # 600 train / 100 val, all unique, seed 42
```

Each row is `{"prompt": "<JSON string>"}`. The MTRL service passes the
`prompt` column to the agent verbatim, so the secret word is packed inside
that string and `parse_task()` in `main.py` unpacks it.

### 3. Deploy the agent

Edit `Wordle/agentcore/aws-targets.json`: replace the placeholder account ID
with yours, and pick the region. The training job must run in the **same
region** as the runtime, or `CreateJob` rejects it.

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
model.

Smoke-test with an RFT-shaped payload (a fake `jobArn` is expected to 400,
which proves the whole chain: payload parsing, bearer-token auth, and error
reporting):

```bash
aws bedrock-agentcore invoke-agent-runtime \
  --agent-runtime-arn <RUNTIME_ARN> \
  --runtime-session-id smoke-test-$(date +%s)000000000000000 \
  --payload fileb://<(echo '{"prompt":"{\"prompt\":\"Guess the 5-letter word\",\"answer\":\"beach\",\"id\":\"t0\"}","metadata":{"jobArn":"arn:aws:sagemaker:us-east-1:111122223333:job/AgentRFT/x","rolloutId":"r1"}}') \
  /dev/stdout
```

### 4. Launch training

```bash
uv run python run_mtrl_training.py \
  --model openai-reasoning-gpt-oss-20b \
  --agent-runtime-arn <RUNTIME_ARN> \
  --role-arn arn:aws:iam::<ACCOUNT>:role/<JOB_ROLE> \
  --s3-prefix s3://<BUCKET>/wordle-mtrl \
  --s3-output-path s3://<BUCKET>/wordle-mtrl/output/ \
  --val-dataset s3://<BUCKET>/wordle-mtrl/validation/validation-data.jsonl \
  --mlflow-app-arn arn:aws:sagemaker:<REGION>:<ACCOUNT>:mlflow-app/<APP_ID> \
  --max-steps 100 --max-epochs 6
```

`--s3-prefix` uploads `training-data.jsonl` for you; pass `--train-dataset`
instead to reuse an existing S3 object. `MlflowConfig` is **required** by
`CreateJob`, so create an MLflow app first
(`aws sagemaker create-mlflow-app ...`) if you don't have one.

Reattach to a running job at any time:

```bash
uv run python run_mtrl_training.py --attach <JOB_NAME> --role-arn ... --s3-output-path ...
```

### 5. Baseline any Bedrock model

```bash
uv run python eval_frontier.py --model global.anthropic.claude-opus-5 --n 100
```

Runs the same environment and reward against a Bedrock model so the number
is directly comparable to MLflow's `val/reward/pass_at_1`.

By default this uses the Converse API. Some models only support tool calling
through Bedrock Mantle, and Claude models on Mantle expose the Anthropic
Messages API rather than Chat Completions, so pick the path per model:

```bash
uv run python eval_frontier.py --model google.gemma-3-27b-it --mantle            # Chat Completions
uv run python eval_frontier.py --model anthropic.claude-opus-5 --mantle messages  # Anthropic Messages
```

Check the model's Bedrock model card for which endpoint and API carry
client-side tool calling.

## IAM

Two roles. Both trust policies matter and one of them is not what the
generic SageMaker docs show.

**Agent runtime role** (created by `agentcore deploy`): trusts
`bedrock-agentcore.amazonaws.com`; needs `AmazonSageMakerJobRuntimeAccess`
(attached by the CDK stack in this repo).

**Training job role** (you create it): must trust
**`job.sagemaker.amazonaws.com`** with both `sts:AssumeRole` and
`sts:TagSession`. Plain `sagemaker.amazonaws.com` is not enough and the error
message says so only after the third retry. Attach
`AmazonSageMakerJobFullAccess`, which covers S3, ECR, EC2 networking, MLflow,
model packages, and `bedrock-agentcore:InvokeAgentRuntime`.

Your caller needs `iam:PassRole` for `job.sagemaker.amazonaws.com` plus the
`sagemaker:*Job*` actions. Full reference:
[MTRL prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl-prereqs.html).

## Hyperparameters

Set in `run_mtrl_training.py`. The ones that mattered:

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | **1e-5** | The documented default for both supported models. The SOP-Bench workshop recipe uses 4e-5; carrying that over collapsed structured tool calling into raw text on Nova by step 15, and produced 4% drift on gpt-oss by step 66. |
| `sampling_max_tokens` | 8192 | The service cap. 4096 truncated gpt-oss mid-reasoning before its first tool call. |
| `temperature` | 1.0 | 1.2 was tried to diversify openers; it wasn't needed and hotter sampling helps a policy wander off its tool-call template. |
| `group_size` | 4 | GRPO group. Reward stdev within groups was reported as 0 at points, so 8–16 is the next thing to try. |
| `global_batch_size` | 32 | 600 prompts → 19 steps/epoch, so 100 steps needs `max_epochs >= 6`. |

`REASONING_EFFORT=low` is set as a runtime env var in `agentcore.json`; both
gpt-oss and Nova 2 honor it as a request parameter. `REASONING_PROMPT_HINT`
additionally prepends `Reasoning: low` to the system prompt, which is the
Harmony convention gpt-oss expects and which Nova should not receive.

## What we learned

Things that cost real training runs to discover, in the order you'll hit them.

**The `@sagemaker_rft_handler` decorator does not start an HTTP server.**
The AWS docs template implies it does. Without `BedrockAgentCoreApp` and
`@app.entrypoint` wrapping it, the runtime times out after 30 s on every
invocation. Decorator order is `@app.entrypoint` outermost.

**`generate_token()` builds a fresh botocore session per call.** Under 32
concurrent rollouts that starves the container credential provider
(`No AWS credentials found`). Caching the token for an hour then failed
differently: it's signed with rotating instance credentials, so cached tokens
go stale and 403 (`Authentication failed`). The fix is one shared botocore
session whose credentials self-refresh, plus a short token cache and a re-sign
on 403.

**Never redeploy the agent while a job is running.** It restarts the
runtime, in-flight rollouts see no sampling requests, and the service fails
the whole job.

**A rollout with zero guesses fails the entire job**
(`No sampling requests were received`). So does returning
`{"status": "error"}` (`The agent signaled that the trajectory failed`).
Always return a reward. The agent retries a zero-guess rollout once with a
fresh conversation, bounded by a wall-clock budget so it can't outrun the
reward-reporting window.

**RL drifts tool calls into plain text.** At 4e-5, Nova went from 10% to
100% raw-text tool calls (`<__function=guess_word>...`) in 90 minutes and
never recovered: no tool executes, every rollout hits the same floor reward,
within-group variance is zero, and GRPO has no gradient back. `main.py`
recovers guesses from raw text (three observed formats, including Gemma's
`guess_word("crane")`) and scores them with a small formatting penalty, which
keeps a gradient alive during drift. This turned out to matter for DeepSeek
and Gemma at eval time too.

**Summing per-guess rewards into one trajectory reward inverts the
incentives.** The original design paid +0.49 for a *lost* game and scored a
turn-1 solve *below* a turn-3 solve, because six turns of base and
exploration bonuses outweighed a −1/turn time penalty. Separating a dominant
outcome term from bounded shaping fixed both. If you change the reward, run
`test_env.py`; it asserts `fast win > slow win > loss > never played`.

**Measure with MLflow, not log scraping.** The service logs
`val/reward/pass_at_1` on the held-out set with `success_threshold = 1.0`,
which for this reward function means "solved." Scraping CloudWatch for
"solved" strings over-reported the win rate by more than 2x because it mixed
training and validation rollouts and dropped rollouts that ended in
exceptions.

**What would improve the 34%.** Validation solve rate peaked at 39% around
step 60 and drifted down, so more steps alone won't help. The most promising
untried change is a curriculum: seed games with 0–4 prior guesses so the
policy learns deduction without first surviving the opening, which
wordle-lora-rl reports as its single biggest gain. Raising `group_size` is
the other obvious lever. Note also that every training row has the identical
prompt text; only the hidden answer varies, so this dataset exercises
"learn from interaction," not "learn from your data."

## Attribution

The environment design, reward constants, clue-state tracking, and word lists
are ported from [charbull/wordle-lora-rl](https://github.com/charbull/wordle-lora-rl)
(MIT), adapted from tag-parsing GRPO to a tool-calling agent on
AgentCore. The optimal-play statistics (3.42 mean turns) follow the
information-theoretic approach popularized by
[3Blue1Brown](https://www.youtube.com/watch?v=v68zYyaEmEA). The pipeline
structure follows the SageMaker MTRL SOP-Bench workshop.

## License

MIT-0 (MIT No Attribution). See [LICENSE](LICENSE).
