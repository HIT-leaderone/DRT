from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_launch_eval_tasks_skips_without_command(monkeypatch):
    module = load_module("launch_eval_task_skip", "verl/trainer/ppo/launch_eval_task.py")
    monkeypatch.delenv("AUTO_EVAL_LAUNCH_COMMAND", raising=False)

    result = module.launch_eval_tasks("prompt", "model", "task", hdfs_path="hdfs://example/path", iter=7)

    assert result == {"status": "skipped", "job_run_id": None, "task": "task", "step": "7"}


def test_launch_eval_tasks_shell_quotes_template_fields(monkeypatch, tmp_path):
    module = load_module("launch_eval_task_quote", "verl/trainer/ppo/launch_eval_task.py")
    output_path = tmp_path / "args with space.json"
    command = (
        f"{shlex.quote(sys.executable)} -c "
        "\"import json, sys; open(sys.argv[1], 'w').write(json.dumps(sys.argv[2:]))\" "
        f"{shlex.quote(str(output_path))} "
        "{task_type} {hdfs_path} {model_type}"
    )
    monkeypatch.setenv("AUTO_EVAL_LAUNCH_COMMAND", command)

    result = module.launch_eval_tasks(
        "prompt",
        "model with space",
        "Task; echo injected",
        hdfs_path="hdfs://example/path with space",
        iter=11,
    )

    assert result["status"] == 0
    assert json.loads(output_path.read_text()) == [
        "Task; echo injected",
        "hdfs://example/path with space",
        "model with space",
    ]


def test_launch_eval_tasks_raises_on_command_error(monkeypatch):
    module = load_module("launch_eval_task_error", "verl/trainer/ppo/launch_eval_task.py")
    monkeypatch.setenv("AUTO_EVAL_LAUNCH_COMMAND", f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit(3)'")

    with pytest.raises(RuntimeError, match="exit code 3"):
        module.launch_eval_tasks("prompt", "model", "task")


def test_monitor_does_not_record_skipped_launch(monkeypatch, tmp_path):
    module = load_module("submit_auto_eval_skip", "submit_auto_eval.py")
    exp = module.Experiment("exp", "model")
    args = SimpleNamespace(
        dry_run=False,
        experiment=["exp"],
        hdfs_root="hdfs://example/root",
        project_name="project",
        judge_model="gpt-4o-2024-05-13",
        openai_api_base="https://api.openai.com/v1",
        openai_auth_scheme="bearer",
        pool="public",
        retry=10,
        tasks="TaskA",
        steps="3",
        min_step=None,
        max_step=None,
        required_file=["config.json"],
        state_file=tmp_path / "state.json",
    )
    state = {"version": 2, "submitted": [], "records": {}}

    monkeypatch.setattr(module, "EXPERIMENTS", (exp,))
    monkeypatch.setattr(module, "list_global_steps", lambda _: [3])
    monkeypatch.setattr(module, "is_checkpoint_ready", lambda *_: True)
    monkeypatch.setattr(
        module,
        "submit_one",
        lambda *_: {"status": "skipped", "job_run_id": None, "task": "TaskA", "step": "3"},
    )

    assert module.scan_once(args, state) == 0
    assert state["records"] == {}
    assert not args.state_file.exists()
