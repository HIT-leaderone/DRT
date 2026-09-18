"""Optional public auto-eval launcher hook.

This module intentionally does not import or call any internal job platform.
Set ``AUTO_EVAL_LAUNCH_COMMAND`` to an executable command template when an
environment wants to submit evaluation work.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from string import Formatter


ALLOWED_FIELDS = {
    "prompt_type",
    "model_type",
    "task_type",
    "hdfs_path",
    "pool",
    "iter",
    "retry",
    "judge_model",
    "openai_api_base",
    "openai_auth_scheme",
}


def _validate_template(template: str) -> None:
    fields = {field for _, field, _, _ in Formatter().parse(template) if field}
    unknown = fields - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unsupported AUTO_EVAL_LAUNCH_COMMAND fields: {sorted(unknown)}")


def _quote_mapping(values: dict[str, object]) -> dict[str, str]:
    return {key: shlex.quote("" if value is None else str(value)) for key, value in values.items()}


def launch_eval_tasks(
    prompt_type,
    model_type,
    task_type,
    hdfs_path=None,
    pool="public",
    iter=0,
    retry=None,
    judge_model=None,
    openai_api_base=None,
    openai_auth_scheme=None,
    openai_api_key=None,
):
    """Launch one optional evaluation task through a user-provided command.

    The command template is read from ``AUTO_EVAL_LAUNCH_COMMAND`` and may use
    these shell-quoted placeholders: ``{prompt_type}``, ``{model_type}``,
    ``{task_type}``, ``{hdfs_path}``, ``{pool}``, ``{iter}``, ``{retry}``,
    ``{judge_model}``, ``{openai_api_base}``, and ``{openai_auth_scheme}``.
    API keys are intentionally passed only via the environment, not templated
    into the command string or printed.
    """

    template = os.environ.get("AUTO_EVAL_LAUNCH_COMMAND")
    payload = {
        "prompt_type": prompt_type,
        "model_type": model_type,
        "task_type": task_type,
        "hdfs_path": hdfs_path,
        "pool": pool,
        "iter": iter,
        "retry": retry,
        "judge_model": judge_model,
        "openai_api_base": openai_api_base,
        "openai_auth_scheme": openai_auth_scheme,
    }

    if not template:
        print(
            "[auto-eval] skipped: AUTO_EVAL_LAUNCH_COMMAND is not set "
            f"(task={task_type}, step={iter})"
        )
        return {"status": "skipped", "job_run_id": None, "task": task_type, "step": str(iter)}

    _validate_template(template)
    command = template.format_map(_quote_mapping(payload))
    env = os.environ.copy()
    if openai_api_key:
        env["AUTO_EVAL_OPENAI_API_KEY"] = str(openai_api_key)

    print(f"[auto-eval] launching task={task_type} step={iter}")
    proc = subprocess.run(command, shell=True, check=False, text=True, env=env)
    result = {"status": proc.returncode, "job_run_id": None, "task": task_type, "step": str(iter)}
    if proc.returncode != 0:
        raise RuntimeError(f"auto-eval command failed with exit code {proc.returncode}")
    return result
