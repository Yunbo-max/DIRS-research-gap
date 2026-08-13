#!/usr/bin/env python3
"""Prophet professional gate runner for the remaining-19 DIRS loop."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet")
RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723/specialized_runners/prophet"
)
PAPER_RUN = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723/paper_runs/"
    "iclr2026_g88nt4ietg_prophet_dlm_early_commit_decoding"
)
DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_text(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    start = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "seconds": round(time.time() - start, 3),
            "stdout": stdout,
            "stderr": stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        else:
            stdout, stderr = "", ""
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "seconds": round(time.time() - start, 3),
            "stdout": stdout or exc.stdout or "",
            "stderr": stderr or exc.stderr or "",
            "timeout": True,
        }


def base_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["HF_HOME"] = "/tf/notebooks/.cache/huggingface"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    return env


def collect_environment(gpu: str) -> dict[str, Any]:
    env = {
        "created_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "repo": str(REPO),
        "dag_path": str(DAG_PATH),
        "selected_cuda_visible_devices": gpu,
        "logical_gpu_mapping_note": (
            f"CUDA_VISIBLE_DEVICES={gpu} makes the selected physical GPU appear as "
            "logical cuda:0 inside PyTorch."
        ),
        "which_python": shutil.which("python"),
        "which_accelerate": shutil.which("accelerate"),
    }
    env["nvidia_smi"] = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    env["python_deps"] = run_command(
        [
            "python",
            "-c",
            (
                "import importlib.util; "
                "mods=['torch','transformers','accelerate','datasets','lm_eval','huggingface_hub']; "
                "print({m: bool(importlib.util.find_spec(m)) for m in mods}); "
                "import torch, transformers, accelerate, datasets, lm_eval, huggingface_hub; "
                "print('torch', torch.__version__, 'cuda', torch.version.cuda); "
                "print('transformers', transformers.__version__); "
                "print('accelerate', accelerate.__version__); "
                "print('datasets', datasets.__version__); "
                "print('lm_eval', getattr(lm_eval, '__version__', 'unknown')); "
                "print('huggingface_hub', huggingface_hub.__version__)"
            ),
        ],
        timeout=60,
    )
    git_dir = REPO / ".git"
    env["git_head_file"] = (
        (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if (git_dir / "HEAD").exists()
        else ""
    )
    env["git_config_file_excerpt"] = (
        compact_text((git_dir / "config").read_text(encoding="utf-8"), 2000)
        if (git_dir / "config").exists()
        else ""
    )
    return env


def list_hf_artifacts() -> dict[str, Any]:
    script = r"""
from huggingface_hub import HfApi
api=HfApi()
for repo_id, repo_type in [
    ('GSAI-ML/LLaDA-8B-Instruct','model'),
    ('Dream-org/Dream-v0-Instruct-7B','model'),
    ('YefanZhou98/DLM-Decoding-Analysis','dataset'),
]:
    try:
        files=api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        print(repo_id, repo_type, len(files))
        for f in files[:30]:
            print(' ', f)
    except Exception as e:
        print(repo_id, repo_type, 'ERR', type(e).__name__, str(e)[:1000])
"""
    result = run_command(["python", "-c", script], cwd=REPO, timeout=120)
    return {
        "artifact_kind": "hf_model_dataset_inventory",
        "command_result": result,
        "convergence_role": "support_only_dependency_resolution",
    }


def run_generation_probe(gpu: str, timeout: int) -> dict[str, Any]:
    worker = RUN_ROOT / "prophet_generation_worker.py"
    result = run_command(
        ["python", str(worker)],
        cwd=REPO,
        env=base_env(gpu),
        timeout=timeout,
    )
    parsed = None
    if result["returncode"] == 0 and result["stdout"].strip():
        try:
            parsed = json.loads(result["stdout"])
        except json.JSONDecodeError as exc:
            marker = '{\n  "artifact_kind"'
            start = result["stdout"].find(marker)
            if start != -1:
                try:
                    parsed = json.loads(result["stdout"][start:])
                except json.JSONDecodeError as exc2:
                    parsed = {
                        "parse_error": repr(exc2),
                        "stdout_tail": compact_text(result["stdout"], 2000),
                    }
            else:
                parsed = {"parse_error": repr(exc), "stdout_tail": compact_text(result["stdout"], 2000)}
    return {
        "artifact_kind": "exact_model_full_parameter_generation_probe",
        "convergence_role": "support_only_not_full_benchmark_grid",
        "command_result": result,
        "parsed": parsed,
    }


def launch_full_gsm8k(gpu: str, variant: str) -> dict[str, Any]:
    assert variant in {"baseline", "prophet"}
    out_dir = RUN_ROOT / f"lm_eval_gsm8k_full_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    pid_path = out_dir / "pid.txt"
    env = base_env(gpu)
    enable = "true" if variant == "prophet" else "false"
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "eval_llada.py",
        "--tasks",
        "gsm8k_cot_zeroshot",
        "--model",
        "llada_dist",
        "--model_args",
        (
            "model_path='GSAI-ML/LLaDA-8B-Instruct',"
            f"enable_early_exit={enable},"
            "constraints_text=\"200:The|201:answer|202:is\","
            "gen_length=256,steps=256,block_length=32,answer_length=5"
        ),
        "--output_path",
        str(out_dir),
    ]
    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
    return {
        "artifact_kind": "full_gsm8k_lm_eval_launch",
        "variant": variant,
        "pid": proc.pid,
        "cmd": cmd,
        "cwd": str(REPO),
        "log_path": str(log_path),
        "output_dir": str(out_dir),
        "cuda_visible_devices": gpu,
        "convergence_role": "running_paper_shaped_eval_not_yet_verifier_accepted",
    }


def poll_full_runs() -> list[dict[str, Any]]:
    rows = []
    for out_dir in sorted(RUN_ROOT.glob("lm_eval_gsm8k_full_*")):
        pid_path = out_dir / "pid.txt"
        log_path = out_dir / "run.log"
        pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
        alive = False
        if pid is not None:
            alive = Path(f"/proc/{pid}").exists()
        log_tail = ""
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            log_tail = text[-6000:]
        output_files = [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()]
        rows.append(
            {
                "variant": out_dir.name.replace("lm_eval_gsm8k_full_", ""),
                "pid": pid,
                "alive": alive,
                "log_path": str(log_path),
                "output_dir": str(out_dir),
                "output_files": output_files,
                "log_tail": log_tail,
            }
        )
    return rows


def verify(
    dag: dict[str, Any],
    hf_inventory: dict[str, Any],
    generation_probe: dict[str, Any] | None,
    full_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    probe_ok = bool(
        generation_probe
        and generation_probe.get("command_result", {}).get("returncode") == 0
        and generation_probe.get("parsed")
    )
    full_running = any(r.get("alive") for r in full_runs)
    full_finished_outputs = [
        r for r in full_runs if (not r.get("alive")) and len(r.get("output_files", [])) > 2
    ]
    unresolved = [
        {
            "id": "full_step_vs_prophet_gsm8k",
            "status": "running" if full_running else "unresolved",
            "needed": "Full GSM8K zero-shot baseline and Prophet lm-eval results with accuracy, steps, and speed.",
        },
        {
            "id": "mmlu_trajectory_analysis",
            "status": "unresolved",
            "needed": "MMLU-STEM trajectory or precomputed DLM-Decoding-Analysis evidence.",
        },
        {
            "id": "multi_benchmark_grid",
            "status": "unresolved",
            "needed": "MMLU, ARC-C, HellaSwag, TruthfulQA, WinoGrande, PIQA, GPQA, HumanEval, MBPP and planning settings.",
        },
        {
            "id": "static_step_and_block_length_ablations",
            "status": "unresolved",
            "needed": "static-step budget ablation and block-length ablation.",
        },
        {
            "id": "dream_model_axis",
            "status": "unresolved",
            "needed": "Dream-7B axis or an explicit code/data blocker for Dream-specific path.",
        },
    ]
    professional_package_ready = bool(full_finished_outputs) and all(
        item["status"] != "unresolved" for item in unresolved
    )
    return {
        "status": (
            "converged_professional_close_match"
            if professional_package_ready
            else (
                "running_full_professional_eval"
                if full_running
                else "blocked_or_partial_after_prophet_specialized_runner"
            )
        ),
        "created_at_utc": utc_now(),
        "blind_contract_checked": dag.get("blind_contract", {}),
        "semantic_dag_nodes_checked": [
            n
            for n in dag.get("nodes", [])
            if n.get("type") in {"gap_hypothesis", "mechanism_binding", "experiment_design"}
        ],
        "hf_inventory_available": hf_inventory.get("command_result", {}).get("returncode") == 0,
        "generation_probe_ok": probe_ok,
        "full_runs": full_runs,
        "unresolved_professional_debt": unresolved,
        "professional_package_ready": professional_package_ready,
        "convergence_decision": "accepted" if professional_package_ready else "not_yet_converged",
        "loop1_required_dag_update": [
            "Add an exact LLaDA-8B model-load/generation gate before full harness launch.",
            "Add full GSM8K baseline-vs-Prophet lm-eval jobs as long-running operational nodes.",
            "Add trajectory-dataset acquisition or local trajectory collection as a separate evidence node.",
            "Add Dream-7B-specific code/data blocker because the local repo snapshot is LLaDA-centered.",
            "Do not accept one-question generation probes as convergence.",
        ],
    }


def write_status(report: dict[str, Any]) -> None:
    lines = [
        "# Prophet Specialized Runner Status",
        "",
        f"- Updated: {utc_now()}",
        f"- Paper: `Diffusion Language Models Know the Answer Before Decoding`",
        f"- Status: `{report['verifier']['status']}`",
        f"- Convergence decision: `{report['verifier']['convergence_decision']}`",
        f"- Generation probe ok: `{report['verifier']['generation_probe_ok']}`",
        f"- Full runs: `{len(report['verifier']['full_runs'])}`",
        f"- Physical GPU requested: `{report['environment']['selected_cuda_visible_devices']}`",
        f"- PyTorch mapping: {report['environment']['logical_gpu_mapping_note']}",
        "",
        "## Full Run State",
    ]
    for row in report["verifier"]["full_runs"]:
        lines.append(
            f"- `{row['variant']}` pid=`{row['pid']}` alive=`{row['alive']}` log=`{row['log_path']}`"
        )
    lines.extend(["", "## Unresolved Professional Debt"])
    for item in report["verifier"]["unresolved_professional_debt"]:
        lines.append(f"- `{item['id']}`: {item['status']} - {item['needed']}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `environment.json`",
            "- `hf_inventory.json`",
            "- `generation_probe.json`",
            "- `prophet_specialized_verifier.json`",
        ]
    )
    (RUN_ROOT / "PROPHET_SPECIALIZED_STATUS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--run-generation-probe", action="store_true")
    parser.add_argument("--generation-timeout", type=int, default=7200)
    parser.add_argument("--launch-full-gsm8k-baseline", action="store_true")
    parser.add_argument("--launch-full-gsm8k-prophet", action="store_true")
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    dag = json.loads(DAG_PATH.read_text(encoding="utf-8"))
    environment = collect_environment(args.gpu)
    hf_inventory = list_hf_artifacts()
    generation_probe = None
    probe_path = RUN_ROOT / "generation_probe.json"
    if args.run_generation_probe:
        generation_probe = run_generation_probe(args.gpu, args.generation_timeout)
    elif probe_path.exists():
        generation_probe = json.loads(probe_path.read_text(encoding="utf-8"))

    launches = []
    if args.launch_full_gsm8k_baseline:
        launches.append(launch_full_gsm8k(args.gpu, "baseline"))
    if args.launch_full_gsm8k_prophet:
        launches.append(launch_full_gsm8k(args.gpu, "prophet"))

    full_runs = poll_full_runs()
    verifier = verify(dag, hf_inventory, generation_probe, full_runs)
    report = {
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "runner_type": "llm_decoding_acceptance_runner",
        "dag_path": str(DAG_PATH),
        "repo": str(REPO),
        "environment": environment,
        "hf_inventory_summary": {
            "returncode": hf_inventory["command_result"]["returncode"],
            "timeout": hf_inventory["command_result"]["timeout"],
            "seconds": hf_inventory["command_result"]["seconds"],
        },
        "generation_probe_summary": {
            "available": generation_probe is not None,
            "returncode": generation_probe.get("command_result", {}).get("returncode")
            if generation_probe
            else None,
            "timeout": generation_probe.get("command_result", {}).get("timeout")
            if generation_probe
            else None,
            "seconds": generation_probe.get("command_result", {}).get("seconds")
            if generation_probe
            else None,
        },
        "launched": launches,
        "verifier": verifier,
    }
    write_json(RUN_ROOT / "environment.json", environment)
    write_json(RUN_ROOT / "hf_inventory.json", hf_inventory)
    if generation_probe is not None:
        write_json(probe_path, generation_probe)
    write_json(RUN_ROOT / "prophet_specialized_verifier.json", report)
    write_status(report)
    print(json.dumps(report["verifier"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
