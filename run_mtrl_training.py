#!/usr/bin/env python3
"""Launch the Wordle MTRL (Agent RFT) training job against the AgentCore-hosted agent.

Adapted from the verified sagemaker-mtrl-workshop launch script
(sopbench-agentcore/scripts/run_rft_training.py). Uses the GA
MultiTurnRLTrainer with agent_env pointed at the deployed Wordle
AgentCore Runtime ARN.

Prereqs:
  1. Deploy the agent and note the runtime ARN (Wordle/agentcore, `agentcore deploy`).
     Redeploying creates a NEW runtime id suffix -- pass --agent-runtime-arn if it changed.
  2. Upload the dataset: this script uploads training-data.jsonl to
     --s3-prefix unless you pass --train-dataset with an existing S3 URI.
  3. IAM:
     - The AgentCore execution role needs the AmazonSageMakerJobRuntimeAccess
       managed policy (added in cdk-stack.ts; redeploy to apply).
     - The training job role (--role-arn) must trust job.sagemaker.amazonaws.com
       (not sagemaker.amazonaws.com) with both sts:AssumeRole and
       sts:TagSession, and allow bedrock-agentcore:GetAgentRuntime + InvokeAgentRuntime on
       runtime/Wordle_Wordle-* (wildcard suffix survives redeploys), S3
       read/write on the dataset/output paths, and sagemaker-mlflow:* on the
       MLflow app if used.
     - Your caller needs iam:PassRole on the job role.
  4. botocore >= 1.43.37 (first version with the SageMaker AgentRFT CreateJob op).

CreateJob MUST run in the SAME region as the AgentCore runtime, so the region
is pinned before any boto3 client is constructed.
"""
import argparse
import inspect
import os
import time

REGION = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = REGION
os.environ["AWS_REGION"] = REGION

from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer  # noqa: E402

DEFAULT_MODEL = "openai-reasoning-gpt-oss-20b"
LOCAL_DATASET = os.path.join(os.path.dirname(__file__), "training-data.jsonl")

# Only three overrides; every other hyperparameter is the gpt-oss-20b default
# (print trainer.hyperparameters.get_info() to see them). The SDK sends only
# values that differ from its defaults, so DescribeJob's JobConfigDocument will
# list just sampling_max_tokens and max_epochs -- max_steps 100 is the default.
HYPERPARAMETERS = {
    # 8192 = service cap (default 4096). At 4096, GPT-OSS-20B often ran out of
    # tokens while reasoning, before its first tool call.
    "sampling_max_tokens": 8192,
    # Training stops at whichever binds first: max_steps, or
    # max_epochs * steps_per_epoch. 640 prompts at the default batch of 128 is
    # exactly 5 steps per epoch, so 20 epochs reach 100 steps.
    "max_epochs": 20,
    "max_steps": 100,
}


def _upload_dataset(s3_prefix: str) -> str:
    """Upload the local JSONL dataset to S3 and return its URI."""
    import boto3

    assert s3_prefix.startswith("s3://"), f"--s3-prefix must be an s3:// URI, got {s3_prefix}"
    bucket, _, key_prefix = s3_prefix[5:].partition("/")
    key = f"{key_prefix.rstrip('/')}/train/training-data.jsonl".lstrip("/")
    boto3.client("s3").upload_file(LOCAL_DATASET, bucket, key)
    uri = f"s3://{bucket}/{key}"
    print(f"Uploaded {LOCAL_DATASET} -> {uri}")
    return uri


def _apply_hyperparameters(trainer, overrides=None):
    hp = trainer.hyperparameters
    print("\n=== applying hyperparameters ===")
    values = dict(HYPERPARAMETERS)
    values.update(overrides or {})
    skipped = []
    for k, v in values.items():
        if hasattr(hp, k):
            setattr(hp, k, v)
            print(f"  set {k} = {v}")
        else:
            skipped.append(k)
            print(f"  SKIP {k} (not on this SDK version)")
    if skipped:
        print(f"\nWARNING: {len(skipped)} hyperparameter(s) not on this SDK: {skipped}. "
              f"Check trainer.hyperparameters.get_info().")


def _build_trainer(args, train_uri: str):
    kwargs = dict(
        model=args.model,
        agent_env=args.agent_runtime_arn,
        training_dataset=train_uri,
        s3_output_path=args.s3_output_path,
        role=args.role_arn,
        accept_eula=True,
    )
    if args.mlflow_app_arn:
        kwargs["mlflow_app_arn"] = args.mlflow_app_arn
    params = inspect.signature(MultiTurnRLTrainer.__init__).parameters
    if args.val_dataset:
        val_param = next((p for p in ("validation_dataset", "eval_dataset", "validation_data")
                          if p in params), None)
        if val_param:
            kwargs[val_param] = args.val_dataset
            print(f"Passing validation via '{val_param}'")
        else:
            print("WARNING: no validation kwarg on this SDK; skipping validation dataset.")
    return MultiTurnRLTrainer(**kwargs)


def _monitor(job, poll: int, timeout: int):
    try:
        print(f"\nMLflow: {job.get_mlflow_url()}")
    except Exception as e:
        print(f"(MLflow URL unavailable: {e})")
    deadline = time.time() + timeout
    terminal = {"Completed", "Failed", "Stopped"}
    while time.time() < deadline:
        job.refresh()
        print(f"  status={job.job_status} secondary={getattr(job, 'secondary_status', '?')}")
        if job.job_status in terminal:
            break
        time.sleep(poll)
    print(f"\nFinal status: {job.job_status}")
    print(f"Output Model Package: {getattr(job, 'output_model_package_arn', None)}")


def main():
    ap = argparse.ArgumentParser(description="Wordle MTRL training job (GA SDK, AgentCore)")
    # Launch-only flags are validated after parsing so --attach can be used alone.
    ap.add_argument("--role-arn",
                    help="Training job IAM role ARN (must trust job.sagemaker.amazonaws.com)")
    ap.add_argument("--s3-output-path", help="s3:// URI for job output")
    ap.add_argument("--s3-prefix", default=None,
                    help="s3:// prefix to upload training-data.jsonl to")
    ap.add_argument("--train-dataset", default=None,
                    help="Existing s3:// URI of the training dataset (skips upload)")
    ap.add_argument("--val-dataset", default=None, help="Optional s3:// validation dataset")
    ap.add_argument("--mlflow-app-arn", default=None, help="Optional MLflow app ARN")
    ap.add_argument("--agent-runtime-arn",
                    help="Deployed Wordle AgentCore runtime ARN (from `agentcore deploy` "
                         "output or Wordle/agentcore/.cli/deployed-state.json)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-steps", type=int, default=None,
                    help=f"Gradient updates (default {HYPERPARAMETERS['max_steps']})")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Passes over the dataset; must cover --max-steps")
    ap.add_argument("--attach", metavar="JOB_NAME", default=None,
                    help="Monitor an existing job instead of launching")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=21600)
    args = ap.parse_args()

    if args.attach:
        _monitor(MultiTurnRLTrainer.attach(job_name=args.attach), args.poll, args.timeout)
        return

    missing = [f for f, v in (("--role-arn", args.role_arn),
                              ("--s3-output-path", args.s3_output_path),
                              ("--agent-runtime-arn", args.agent_runtime_arn)) if not v]
    if missing:
        ap.error(f"launching a job requires {', '.join(missing)} (or use --attach JOB_NAME)")

    supported = MultiTurnRLTrainer.list_supported_models()
    assert args.model in supported, f"{args.model} not in supported models: {supported}"

    if args.train_dataset:
        train_uri = args.train_dataset
    elif args.s3_prefix:
        train_uri = _upload_dataset(args.s3_prefix)
    else:
        ap.error("Provide either --train-dataset (existing S3 URI) or --s3-prefix (to upload)")

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs

    trainer = _build_trainer(args, train_uri)
    _apply_hyperparameters(trainer, overrides)

    print("\nSubmitting job (wait=False) ...")
    job = trainer.train(wait=False)
    print(f"Job: {getattr(job, 'job_name', '?')}  Status: {job.job_status}")
    _monitor(job, args.poll, args.timeout)


if __name__ == "__main__":
    main()
