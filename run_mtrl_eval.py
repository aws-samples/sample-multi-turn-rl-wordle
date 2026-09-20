#!/usr/bin/env python3
"""Evaluate the base and/or MTRL-trained model with a SageMaker evaluation job.

Evaluation jobs (JobCategory=AgentRFTEvaluation) play the held-out words
through the SAME AgentCore runtime and reward used for training, on either
the base model or a trained model package, and log eval/reward/* metrics to
MLflow. Unlike the validation pass inside a training job, you control the
sampling temperature here, so it is the right tool for questions like
"does the adapter still help at a different reasoning effort or temperature?"

Reasoning effort is NOT a job parameter: the agent reads REASONING_EFFORT
from agentcore.json. To evaluate at a different effort, change that value,
`agentcore deploy`, run the eval, then deploy the training value back.
Never redeploy while a training or evaluation job is running.

Only one evaluation job runs at a time per account (default quota), so
multiple --runs execute sequentially.

Examples:
  # base vs trained, greedy, current runtime effort
  uv run python run_mtrl_eval.py --runs base:0 trained:0 \
      --model-package-arn arn:aws:sagemaker:...:model-package/<group>/<version> \
      --role-arn arn:aws:iam::<ACCOUNT>:role/<JOB_ROLE> \
      --agent-runtime-arn <RUNTIME_ARN> \
      --dataset s3://<BUCKET>/wordle-mtrl/validation/validation-data.jsonl \
      --s3-output-path s3://<BUCKET>/wordle-mtrl/eval/ \
      --mlflow-app-arn arn:aws:sagemaker:<REGION>:<ACCOUNT>:mlflow-app/<APP_ID>

  # attach to a running eval job
  uv run python run_mtrl_eval.py --attach wordle-eval-base-t0-20260920...
"""
import argparse
import json
import os
import re
import time

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = REGION
os.environ["AWS_REGION"] = REGION

import boto3  # noqa: E402

DEFAULT_MODEL = "openai-reasoning-gpt-oss-20b"
TERMINAL = {"Completed", "Failed", "Stopped"}


def _base_model_arn(model: str) -> str:
    return f"arn:aws:sagemaker:{REGION}:aws:hub-content/SageMakerPublicHub/Model/{model}"


def _job_name(which: str, temperature: float, tag: str) -> str:
    t = f"t{temperature}".replace(".", "p")
    stamp = time.strftime("%Y%m%d%H%M%S")
    name = f"wordle-eval-{which}-{t}{('-' + tag) if tag else ''}-{stamp}"
    return re.sub(r"[^a-zA-Z0-9-]", "-", name)[:63]


def _job_config(args, which: str, temperature: float, run_name: str) -> dict:
    doc = {
        "AgentConfig": {
            "BedrockAgentCoreConfig": {
                "AgentRuntimeArn": args.agent_runtime_arn,
                "Qualifier": "DEFAULT",
            }
        },
        "InputDataConfig": [{
            "ChannelName": "evaluation",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": args.dataset}},
        }],
        "OutputDataConfig": {
            "S3OutputPath": args.s3_output_path,
            "MlflowConfig": {
                "MlflowResourceArn": args.mlflow_app_arn,
                "MlflowExperimentName": args.mlflow_experiment,
                "MlflowRunName": run_name,
            },
        },
        "EvaluationConfig": {
            "BaseModelArn": _base_model_arn(args.model),
            "AcceptEula": True,
            # Flat string values. These are the names the CreateJob validator
            # accepts (it lists them in its error message); the nested form in
            # the docs and the SDK's sampling_temperature/top_p/max_tokens are
            # both rejected.
            "HyperParameters": {
                "eval_group_size": str(args.group_size),
                "temperature": str(temperature),
                "sampling_top_p": "1.0",
                # Training used 8192; the eval default of 4096 truncates
                # gpt-oss mid-reasoning at medium/high effort.
                "sampling_max_tokens": str(args.max_tokens),
                "pass_k_values": json.dumps([k for k in (1, 2, 4, 8) if k <= args.group_size]),
                "success_threshold": "1",       # reward >= 1.0 means "solved"
                "rollout_timeout": "600",
                "rollout_max_concurrency": str(args.concurrency),
                "rollout_max_retries": "3",
            },
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.timeout},
    }
    if which == "trained":
        doc["ModelPackageConfig"] = {"InputModelPackageArn": args.model_package_arn}
    return doc


def _describe(sm, name: str) -> dict:
    return sm.describe_job(JobName=name, JobCategory="AgentRFTEvaluation")


def _wait(sm, name: str, poll: int, timeout: int) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        d = _describe(sm, name)
        cur = (d["JobStatus"], d.get("SecondaryStatus"))
        if cur != last:
            print(f"  [{time.strftime('%H:%M:%S')}] status={cur[0]} secondary={cur[1]}")
            last = cur
        if d["JobStatus"] in TERMINAL:
            return d
        time.sleep(poll)
    raise TimeoutError(f"{name} still running after {timeout}s")


def _summarize(d: dict):
    doc = d.get("JobConfigDocument")
    doc = json.loads(doc) if isinstance(doc, str) else (doc or {})
    out = doc.get("ServiceOutput", {})
    print(f"\n=== {d['JobName']}: {d['JobStatus']} ===")
    if d["JobStatus"] != "Completed":
        print(json.dumps({k: v for k, v in d.items() if "Failure" in k or "Reason" in k},
                         indent=1, default=str))
    for k in ("RolloutInfo", "BillableTokenUsage", "MlflowDetails", "EvaluationMetrics"):
        if k in out:
            print(f"{k}: {json.dumps(out[k], default=str)}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Wordle MTRL evaluation job(s) (AgentCore)")
    ap.add_argument("--runs", nargs="+", default=["base:0", "trained:0"],
                    help="which:temperature pairs, e.g. base:0 trained:0 base:1.0 trained:1.0")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base hub model name")
    ap.add_argument("--model-package-arn", help="trained model package ARN (for 'trained' runs)")
    ap.add_argument("--role-arn", help="job IAM role (must trust job.sagemaker.amazonaws.com)")
    ap.add_argument("--agent-runtime-arn", help="deployed Wordle AgentCore runtime ARN")
    ap.add_argument("--dataset", help="s3:// URI of the held-out evaluation JSONL")
    ap.add_argument("--s3-output-path", help="s3:// prefix for evaluation output")
    ap.add_argument("--mlflow-app-arn", help="MLflow app ARN (metrics land under eval/)")
    ap.add_argument("--mlflow-experiment", default="wordle-mtrl-eval")
    ap.add_argument("--group-size", type=int, default=1, help="rollouts per prompt")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32, help="parallel rollouts")
    ap.add_argument("--tag", default="", help="extra token for the job name, e.g. 'medium'")
    ap.add_argument("--no-wait", action="store_true", help="submit the first run and exit")
    ap.add_argument("--attach", metavar="JOB_NAME", help="monitor an existing eval job")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=4 * 3600, help="per job, seconds")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)

    if args.attach:
        _summarize(_wait(sm, args.attach, args.poll, args.timeout))
        return

    required = ["role_arn", "agent_runtime_arn", "dataset", "s3_output_path", "mlflow_app_arn"]
    missing = [f"--{r.replace('_', '-')}" for r in required if not getattr(args, r)]
    if missing:
        ap.error(f"launching requires {', '.join(missing)}")

    runs = []
    for r in args.runs:
        which, _, temp = r.partition(":")
        if which not in ("base", "trained"):
            ap.error(f"run '{r}': expected base:<temp> or trained:<temp>")
        if which == "trained" and not args.model_package_arn:
            ap.error("'trained' runs need --model-package-arn")
        runs.append((which, float(temp or 0)))

    results = []
    for i, (which, temp) in enumerate(runs):
        name = _job_name(which, temp, args.tag)
        doc = _job_config(args, which, temp, run_name=name)
        print(f"\n[{i + 1}/{len(runs)}] creating {name}  (temperature={temp})")
        resp = sm.create_job(
            JobName=name,
            JobCategory="AgentRFTEvaluation",
            RoleArn=args.role_arn,
            JobConfigSchemaVersion="1.0.0",
            JobConfigDocument=json.dumps(doc),
        )
        print(f"  {resp['JobArn']}")
        if args.no_wait:
            return
        d = _wait(sm, name, args.poll, args.timeout)
        results.append((which, temp, _summarize(d)))

    print("\n=== summary ===")
    for which, temp, out in results:
        print(f"{which:8s} temp={temp}: {json.dumps(out.get('EvaluationMetrics', out), default=str)[:300]}")


if __name__ == "__main__":
    main()
