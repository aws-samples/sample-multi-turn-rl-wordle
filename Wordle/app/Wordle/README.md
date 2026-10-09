# Wordle agent (AgentCore runtime)

This directory is what gets deployed to Amazon Bedrock AgentCore. See the
[repository README](../../../README.md) for the full picture; this file covers
only what lives here.

## Files

| File | Role |
|---|---|
| `main.py` | Everything the runtime needs: the `WordleGame` environment and reward, the Strands agent with its single `guess_word` tool, the RFT rollout handler, bearer-token handling, and raw-text tool-call recovery. |
| `test_env.py` | Offline tests: Wordle scoring, clue tracking, reward ordering (`fast win > slow win > loss > never played`), and a round-trip of every dataset row through `parse_task()`. No AWS calls. |
| `data/` | `answers.txt` (1,923 common words that can be the secret), `allowed_guesses.txt` (11,072 words accepted as guesses), and `word_entropy.json`, the per-guess entropy used for the opening-guess bonus. All three are generated from [SCOWL](http://wordlist.aspell.net/) by `build_word_lists.py` at the repository root; don't edit them by hand. |

## Entrypoint contract

```python
app = BedrockAgentCoreApp()

@app.entrypoint          # outermost: serves HTTP on :8080
@sagemaker_rft_handler   # inner: reports CompleteRollout + UpdateReward
def handle_rollout(payload): ...
```

The handler receives `{"prompt": "<JSON string>", "metadata": {...},
"inferenceParams": {...}}`, plays one game against the policy model at
`metadata.endpoint`, and returns `{"reward": <float>}`. It always returns a
reward; it never returns `{"status": "error"}`, because the training service
treats that as a fatal trajectory failure.

## Environment variables

Set in `../../agentcore/agentcore.json` under `envVars`.

| Variable | Default | Purpose |
|---|---|---|
| `REASONING_EFFORT` | `low` | Sent as `reasoning_effort` on every inference request. `low`/`medium`/`high` enable it; anything else disables it. Honored by gpt-oss and Nova 2. |
| `REASONING_PROMPT_HINT` | `false` | Also prepend `Reasoning: <level>` to the system prompt (gpt-oss Harmony convention; do not enable for Nova). |
| `ROLLOUT_BUDGET_SECONDS` | `420` | Wall-clock cap on retries so a rollout can't outrun the service's reward-reporting window (`rollout_timeout`, 600 s). |
| `RFT_RUNTIME_ENDPOINT` | regional default | Fallback if `metadata.endpoint` is absent from the payload. |
| `AWS_REGION` | `us-east-1` | Region for bearer-token signing and the endpoint fallback. |

## Local testing

```bash
uv sync
uv run python test_env.py
```

To confirm the HTTP server boots without deploying:

```bash
uv run python -c "
import threading, time, urllib.request, main
threading.Thread(target=main.app.run, kwargs={'port': 8181}, daemon=True).start()
time.sleep(3)
print(urllib.request.urlopen('http://127.0.0.1:8181/ping').read())"
```
