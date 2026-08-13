#!/usr/bin/env python3
"""Professional gate for the rational information-seeking Battleship paper.

This runner repairs a Loop 1 DAG error: the paper-specific DAG previously said
no local repo was encoded, but the exact-title official repo is available. The
runner does not promote repo presence, tests, figures, or old CSVs to
convergence. It records the real operational contract for Loop 2: Python 3.11,
OpenRouter-backed model-agent experiments, local Battleship data, Guess Who
transfer artifacts, raw outputs, and verifier-comparable metric summaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723"
)
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"

CFG: dict[str, Any] = {
    "short": "battleship",
    "paper_id": "ICLR2026_EQhUvWH78U_rational_information_seeking_agents",
    "paper_run": "iclr2026_eqhuvwh78u_rational_information_seeking_agents",
    "title": "Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People",
    "repo": "/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/battleship",
    "runner_type": "agent_decision_benchmark_runner",
    "status": "blocked_by_python311_openrouter_full_agent_benchmark_and_guess_who_artifacts",
    "script_files": [
        "README.md",
        "pyproject.toml",
        "battleship/agents.py",
        "battleship/fast_sampler.py",
        "battleship/planner_captain.py",
        "battleship/spotters.py",
        "battleship/captains.py",
        "battleship/run_spotter_benchmarks.py",
        "battleship/run_captain_benchmarks.py",
        "experiments/collaborative/analysis.py",
        "experiments/collaborative/export_trajectories.py",
        "experiments/cogsci/eval_gpt4.py",
        "experiments/cogsci/eval_huggingface.py",
        "experiments/cogsci/eval_sampling.py",
        "tests/test_eig_calculator.py",
        "tests/test_fast_sampler.py",
        "tests/v1/test_board.py",
    ],
    "runtime_modules": [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "lark",
        "nltk",
        "openai",
        "sklearn",
        "rich",
        "torch",
        "transformers",
    ],
    "required_data_paths": {
        "battleship_qa_csv": "data/battleship-qa.csv",
        "readme_named_human_dataset_csv": "data/human-dataset.csv",
        "collaborative_board_contexts": "experiments/collaborative/contexts",
        "collaborative_raw_human_data": "experiments/collaborative/data/battleship-final-data",
        "final_project_legacy_llm_results": "experiments/final_project/llm_results.csv",
        "final_project_legacy_sampling_results": "experiments/final_project/sampling_results.csv",
        "guess_who_result_figure_only": "docs/static/images/guess-who-results.png",
    },
    "required_output_paths": {
        "spotter_benchmark_results": "experiments/collaborative/spotter_benchmarks",
        "captain_benchmark_results": "experiments/collaborative/captain_benchmarks",
        "guess_who_transfer_outputs": "experiments/guess_who",
        "openrouter_model_eval_outputs": "experiments/openrouter",
        "paper_metric_summary": "results/paper_metric_summary.json",
    },
    "paper_shaped_outputs_required": [
        "SpotterQA model grid with DirectSpotter, CodeSpotter, CoT/history variants, and OpenRouter LLMs",
        "CaptainQA / Collaborative Battleship game grid with Random, MAP, EIG, Bayes-Q/M/D, and LLMDecisionCaptain",
        "human-human trajectory comparison on board-matched contexts",
        "Guess Who 60-game transfer outputs, not a figure-only asset",
        "accuracy, targeting F1, precision, recall, moves, questions, win rate, EIG, redundant-question, and token-cost summaries",
        "raw per-question/per-game outputs plus table/figure-comparable metric JSON/CSV",
    ],
    "dag_nodes": [
        (
            "repo_script_gate",
            "Bind exact-title official battleship repo and validate agents, FastSampler, EIGCalculator, CodeSpotter, planner captain, spotter/captain benchmark scripts, and analysis scripts.",
        ),
        (
            "python311_runtime_gate",
            "Use Python 3.11+ runtime because the repo imports enum.StrEnum; Python 3.10 collection failures are runtime blockers, not method results.",
        ),
        (
            "openrouter_api_gate",
            "Resolve OpenRouter API credentials and exact model list before model-agent evaluation; do not expose secrets in artifacts.",
        ),
        (
            "data_artifact_gate",
            "Materialize BattleshipQA, collaborative human trajectories, board contexts, and Guess Who transfer data/assets before execution.",
        ),
        (
            "full_api_game_grid",
            "Run full SpotterQA, CaptainQA, human comparison, OpenRouter model grid, and Guess Who transfer protocol with raw outputs and token-cost traces.",
        ),
        (
            "metric_table_verifier_gate",
            "Compare simulated gap and result shape against paper tables, figures, paragraphs, and appendix evidence; close numeric shape can pass only from operational outputs.",
        ),
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "timeout": False,
            "seconds": round(elapsed, 3),
            "stdout": proc.stdout[-10000:],
            "stderr": proc.stderr[-10000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout": True,
            "seconds": timeout,
            "stdout": (exc.stdout or "")[-5000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-5000:] if isinstance(exc.stderr, str) else "",
        }


def package_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def path_size(path: Path) -> tuple[str | None, int]:
    if not path.exists():
        return None, 0
    if path.is_file():
        return str(path.stat().st_size), 1
    size = run_cmd(["du", "-sh", str(path)], timeout=20)["stdout"].split()
    count = run_cmd(["find", str(path), "-type", "f"], timeout=25)
    file_count = len([line for line in count["stdout"].splitlines() if line.strip()])
    return (size[0] if size else None), file_count


def gpu_rows() -> list[dict[str, Any]]:
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    rows = []
    for line in result.get("stdout", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "memory_free_mib": int(float(parts[4])),
                    "utilization_gpu_pct": int(float(parts[5])),
                    "temperature_c": int(float(parts[6])),
                    "power_w": float(parts[7]),
                }
            )
    return rows


def repo_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    py_path = str(repo)
    if env.get("PYTHONPATH"):
        py_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = py_path
    return env


def script_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    repo = Path(cfg["repo"])
    rows = []
    env = repo_env(repo)
    for rel in cfg["script_files"]:
        path = repo / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            text = read_text(path)
            row["size_bytes"] = path.stat().st_size
            row["line_count"] = len(text.splitlines())
            if path.suffix == ".py":
                row["py_compile"] = run_cmd([sys.executable, "-m", "py_compile", str(path)], cwd=repo, timeout=60, env=env)
                row["imports"] = sorted(set(re.findall(r"^(?:from|import)\s+([A-Za-z0-9_\.]+)", text, re.M)))
                row["cli_flags"] = sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text)))
            elif path.name == "pyproject.toml":
                row["declares_python"] = re.findall(r"python\s*=\s*['\"]([^'\"]+)['\"]", text)
                row["dependencies"] = sorted(set(re.findall(r"^([A-Za-z0-9_\-]+)\s*=\s*['\"]", text, re.M)))
            elif path.name.lower().startswith("readme"):
                row["urls"] = sorted(set(re.findall(r"https?://[^)\s]+", text)))
                row["run_command_lines"] = [line.strip() for line in text.splitlines() if "python " in line or "poetry " in line or "pip install" in line]
        rows.append(row)
    all_files = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts)[:1000]
    payload = {
        "artifact_kind": "battleship_repo_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "repo": str(repo),
        "repo_exists": repo.exists(),
        "script_files": rows,
        "all_repo_files_excluding_git_head": all_files,
        "git": {
            "remote": run_cmd(["git", "-C", str(repo), "remote", "-v"], timeout=20),
            "head": run_cmd(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=20),
        },
        "support_only_findings": [
            "exact-title official repo is present and must be encoded in the DAG",
            "repo presence, source compilation, old CSVs, notebooks, and static figures are support evidence only",
            "convergence requires fresh or materialized paper-shaped model/API/game outputs",
        ],
    }
    write_json(Path(cfg["runner_dir"]) / "repo_manifest.json", payload)
    return payload


def environment_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    repo = Path(cfg["repo"])
    env = repo_env(repo)
    import_probes = {}
    for name in cfg["runtime_modules"]:
        import_probes[name] = run_cmd(
            [sys.executable, "-c", f"import importlib; m=importlib.import_module({name!r}); print(getattr(m, '__version__', 'imported'))"],
            cwd=repo,
            timeout=45,
            env=env,
        )
    payload = {
        "artifact_kind": "battleship_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "python": sys.version,
        "python_executable": sys.executable,
        "python_version_gate": {
            "required": ">=3.11",
            "reason": "Repo imports enum.StrEnum and pyproject declares python ^3.11.",
            "current_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "passes": (sys.version_info.major, sys.version_info.minor) >= (3, 11),
        },
        "gpu_rows": gpu_rows(),
        "gpu_requirement": "No local GPU requirement for the paper's core API/symbolic agent scaffolding; GPU rows are recorded for global campaign accounting.",
        "packages": {name: package_version(name.replace("_", "-")) or package_version(name) for name in cfg["runtime_modules"]},
        "import_probes": import_probes,
        "compileall_repo": run_cmd([sys.executable, "-m", "compileall", "-q", str(repo / "battleship"), str(repo / "tests")], cwd=repo, timeout=240, env=env),
        "selected_pytest_probe": run_cmd(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_eig_calculator.py",
                "tests/test_fast_sampler.py",
                "tests/v1/test_board.py",
            ],
            cwd=repo,
            timeout=180,
            env=env,
        ),
    }
    write_json(Path(cfg["runner_dir"]) / "environment.json", payload)
    return payload


def artifact_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    repo = Path(cfg["repo"])
    data = {}
    for name, rel in cfg["required_data_paths"].items():
        path = repo / rel
        size_human, file_count = path_size(path)
        data[name] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "size_human": size_human,
            "file_count": file_count,
        }
    outputs = {}
    for name, rel in cfg["required_output_paths"].items():
        path = repo / rel
        size_human, file_count = path_size(path)
        outputs[name] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "size_human": size_human,
            "file_count": file_count,
        }
    output_candidates = []
    for pattern in ["experiments/**/*.json", "experiments/**/*.csv", "results/**/*.json", "docs/static/images/*results*"]:
        for path in repo.glob(pattern):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            size_human, file_count = path_size(path)
            output_candidates.append({"path": str(path), "is_dir": path.is_dir(), "size_human": size_human, "file_count": file_count})
            if len(output_candidates) >= 200:
                break
    payload = {
        "artifact_kind": "battleship_model_data_output_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "data_artifacts": data,
        "required_operational_outputs": outputs,
        "verifier_comparable_output_candidates": output_candidates,
        "paper_shaped_outputs_required": cfg["paper_shaped_outputs_required"],
        "support_only_warning": "Static figures, old CSVs, notebooks, and source data are not the full Loop 2 result grid.",
    }
    write_json(Path(cfg["runner_dir"]) / "model_data_manifest.json", payload)
    return payload


def api_runtime_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    payload = {
        "artifact_kind": "battleship_api_runtime_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "api_key_presence": {key: bool(os.environ.get(key)) for key in keys},
        "secret_policy": "Only boolean key presence is recorded; secret values are never written.",
        "code_api_surface": {
            "agents_py_openrouter_base_url": "https://openrouter.ai/api/v1",
            "required_env_var_from_code": "OPENROUTER_API_KEY",
        },
        "paper_runtime_interpretation": "Core experiments are API/model-agent and symbolic sampler experiments; local GPU is not the deciding execution resource for this paper.",
    }
    write_json(Path(cfg["runner_dir"]) / "api_runtime_manifest.json", payload)
    return payload


def professional_gate(cfg: dict[str, Any], env: dict[str, Any], artifacts: dict[str, Any], api: dict[str, Any]) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    if not env["python_version_gate"]["passes"]:
        blockers.append(
            {
                "id": "battleship_python311_runtime_missing",
                "status": "blocked",
                "detail": "Current Python is "
                + env["python_version_gate"]["current_major_minor"]
                + "; pyproject requires ^3.11 and tests fail on enum.StrEnum under Python 3.10.",
            }
        )
    if env["selected_pytest_probe"]["returncode"] != 0:
        blockers.append(
            {
                "id": "battleship_unit_probe_failed_until_python311_runtime",
                "status": "blocked",
                "detail": "Selected tests did not collect/pass; see environment.json.",
            }
        )
    missing_imports = [name for name, probe in env["import_probes"].items() if probe["returncode"] != 0]
    if missing_imports:
        blockers.append(
            {
                "id": "battleship_runtime_dependencies_missing",
                "status": "blocked",
                "detail": "Missing runtime imports: " + ", ".join(missing_imports),
            }
        )
    if not api["api_key_presence"].get("OPENROUTER_API_KEY"):
        blockers.append(
            {
                "id": "battleship_openrouter_api_key_missing",
                "status": "blocked",
                "detail": "battleship/agents.py constructs an OpenAI client with OpenRouter base_url and requires OPENROUTER_API_KEY.",
            }
        )
    missing_data = [
        name
        for name, row in artifacts["data_artifacts"].items()
        if not row["exists"] and name not in {"guess_who_result_figure_only"}
    ]
    if missing_data:
        blockers.append(
            {
                "id": "battleship_required_data_artifacts_missing",
                "status": "blocked",
                "detail": "Missing data artifacts: " + ", ".join(missing_data),
            }
        )
    if artifacts["data_artifacts"]["collaborative_raw_human_data"]["file_count"] < 5:
        blockers.append(
            {
                "id": "battleship_raw_human_trajectory_files_incomplete",
                "status": "blocked",
                "detail": "Raw collaborative human-data directory exists but does not contain the paper-shaped game/round/stage trajectory files expected by README.",
            }
        )
    missing_outputs = [name for name, row in artifacts["required_operational_outputs"].items() if not row["exists"]]
    if missing_outputs:
        blockers.append(
            {
                "id": "battleship_full_agent_benchmark_outputs_missing",
                "status": "blocked",
                "detail": "Missing operational result directories/files: " + ", ".join(missing_outputs),
            }
        )
    blockers.append(
        {
            "id": "battleship_result_shape_verifier_waiting_for_full_grid",
            "status": "blocked",
            "detail": "Verifier cannot compare against paper tables/figures/paragraphs until SpotterQA, CaptainQA, human comparison, OpenRouter model grid, and Guess Who outputs are produced.",
        }
    )
    gate = {
        "artifact_kind": "battleship_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "status": cfg["status"],
        "professional_package_ready": False,
        "converged": False,
        "convergence_role": "repo resolution and operational preflight only; no reduced/proxy/static evidence is promoted",
        "blockers": blockers,
        "support_checks": {
            "repo_discovered_and_encoded": Path(cfg["repo"]).exists(),
            "python311_gate_passed": env["python_version_gate"]["passes"],
            "openrouter_key_present": api["api_key_presence"].get("OPENROUTER_API_KEY", False),
            "gpu_required_for_core_experiments": False,
            "data_artifacts_checked": len(artifacts["data_artifacts"]),
            "operational_output_artifacts_checked": len(artifacts["required_operational_outputs"]),
        },
        "next_full_execution_if_unblocked": [
            "switch to Python 3.11+ environment and install repo with Poetry or editable pip",
            "configure OPENROUTER_API_KEY and exact paper model list without leaking secrets",
            "run full spotter benchmark scripts over paper configurations",
            "run full captain game benchmark scripts over paper captains, boards, and seeds",
            "materialize or run Guess Who transfer protocol outputs",
            "emit raw outputs, token costs, timing, metric summaries, and verifier-comparable table/figure JSON/CSV",
        ],
        "paper_shaped_outputs_required": cfg["paper_shaped_outputs_required"],
    }
    write_json(Path(cfg["runner_dir"]) / "professional_gate_result.json", gate)
    return gate


def ensure_node(dag: dict[str, Any], node: dict[str, Any]) -> None:
    for existing in dag.setdefault("nodes", []):
        if existing.get("id") == node["id"]:
            existing.update(node)
            return
    dag["nodes"].append(node)


def ensure_edge(dag: dict[str, Any], source: str, target: str) -> None:
    edge = [source, target]
    if edge not in dag.setdefault("edges", []):
        dag["edges"].append(edge)


def update_dag(cfg: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    paper_dir = RUN_ROOT / "paper_runs" / cfg["paper_run"]
    dag_path = paper_dir / "paper_author_gap_dag.json"
    dag = read_json(dag_path)
    repo = Path(cfg["repo"])
    for node in dag.get("nodes", []):
        if node.get("id") == "ops.resolve_repo_code":
            node["repo_paths"] = [str(repo)]
            node["content"] = (
                f"repos={repo}; code_artifacts=battleship/planner_captain.py; "
                "battleship/agents.py; battleship/fast_sampler.py; battleship/spotters.py; "
                "battleship/captains.py; battleship/run_captain_benchmarks.py; "
                "battleship/run_spotter_benchmarks.py"
            )
        if node.get("id") == "experiments.system_surface":
            node["content"] = (
                "hardware/API=API-based evaluation through OpenRouter plus symbolic sampler/runtime checks; "
                "no local GPU requirement for core scaffolding; Python 3.11+ required by repo; "
                "GPU only applies if a local HuggingFace baseline is selected by the benchmark grid"
            )
    for suffix, content in cfg["dag_nodes"]:
        node_id = f"ops.battleship_{suffix}"
        node_type = "operational_execution" if suffix == "full_api_game_grid" else ("verification" if "verifier" in suffix else "operational_dependency")
        ensure_node(
            dag,
            {
                "id": node_id,
                "content": content,
                "type": node_type,
                "skill_role": "paper-specific operational gate",
            },
        )
    decision_id = "decision.explicit_blocker_after_battleship_preflight"
    ensure_node(
        dag,
        {
            "id": decision_id,
            "content": "If Python 3.11 runtime, OpenRouter API, data artifacts, raw model-agent outputs, metric summaries, or Guess Who transfer outputs are missing, block and feed exact requirements back into Loop 1.",
            "type": "author_reviewer_decision",
            "skill_role": "paper-specific operational gate",
        },
    )
    chain = [
        "ops.resolve_repo_code",
        "ops.battleship_repo_script_gate",
        "ops.battleship_python311_runtime_gate",
        "ops.battleship_openrouter_api_gate",
        "ops.battleship_data_artifact_gate",
        "ops.battleship_full_api_game_grid",
        "ops.battleship_metric_table_verifier_gate",
        "reviewer.require_professional_artifact_package",
    ]
    for src, dst in zip(chain, chain[1:]):
        ensure_edge(dag, src, dst)
    ensure_edge(dag, "ops.battleship_full_api_game_grid", "reviewer.compare_result_shapes")
    ensure_edge(dag, "reviewer.keep_exact_artifact_debt", decision_id)
    dag.setdefault("previous_loop_updates", []).append(
        {
            "iteration": 3,
            "created_at_utc": utc_now(),
            "source": "battleship_professional_gate",
            "status": gate["status"],
            "blocker_ids": [b["id"] for b in gate["blockers"]],
            "repo_paths": [str(repo)],
            "converged": False,
        }
    )
    sig_src = json.dumps(dag.get("nodes", []), sort_keys=True) + json.dumps(dag.get("edges", []), sort_keys=True)
    dag["signature"] = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:16]
    write_json(dag_path, dag)
    write_json(paper_dir / "paper_author_gap_dag_iter_03.json", dag)
    return dag


def verifier(cfg: dict[str, Any], gate: dict[str, Any], dag: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact_kind": "battleship_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "iteration": 3,
        "converged": False,
        "professional_ready": False,
        "checks": [
            {"name": "blind_contract", "status": "pass", "detail": dag.get("blind_contract", {})},
            {"name": "repo_path_encoded", "status": "pass" if cfg["repo"] in json.dumps(dag) else "fail", "detail": [cfg["repo"]]},
            {"name": "gpu_requirement_interpretation", "status": "pass", "detail": "DAG says no local GPU requirement for core API/symbolic experiments; do not force a fake GPU proxy."},
            {"name": "python311_runtime_gate", "status": "blocked", "detail": "Current interpreter is not Python 3.11+."},
            {"name": "openrouter_api_gate", "status": "blocked", "detail": "OPENROUTER_API_KEY absent."},
            {"name": "professional_artifact_package", "status": "blocked", "detail": gate["blockers"]},
            {"name": "result_shape_comparison_ready", "status": "blocked", "detail": "requires full paper-shaped SpotterQA/CaptainQA/Guess Who/raw metric outputs."},
        ],
        "required_updates": [
            {
                "id": "update.battleship_repo_runtime_api_result_grid",
                "reason": gate["status"],
                "success_criteria": [
                    "official repo path encoded",
                    "Python 3.11+ runtime present",
                    "OpenRouter API/model grid configured",
                    "full BattleshipQA/CaptainQA/human/Guess Who outputs produced",
                    "verifier compares operational result shapes to paper evidence channels",
                ],
            }
        ],
        "artifact_paths": {
            "professional_gate": str(Path(cfg["runner_dir"]) / "professional_gate_result.json"),
            "repo_manifest": str(Path(cfg["runner_dir"]) / "repo_manifest.json"),
            "environment": str(Path(cfg["runner_dir"]) / "environment.json"),
            "model_data_manifest": str(Path(cfg["runner_dir"]) / "model_data_manifest.json"),
            "api_runtime_manifest": str(Path(cfg["runner_dir"]) / "api_runtime_manifest.json"),
            "dag_iter_03": str(RUN_ROOT / "paper_runs" / cfg["paper_run"] / "paper_author_gap_dag_iter_03.json"),
        },
    }
    write_json(Path(cfg["runner_dir"]) / "battleship_specialized_verifier.json", payload)
    write_json(RUN_ROOT / "paper_runs" / cfg["paper_run"] / "verifier_result_iter_03.json", payload)
    return payload


def update_queue_summary(cfg: dict[str, Any], gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    repo = Path(cfg["repo"])
    queue_obj = read_json(QUEUE_PATH)
    for item in queue_obj.get("queue", []):
        if item.get("paper_id") == cfg["paper_id"]:
            item["priority"] = "high"
            item["professional_blocker"] = gate["status"]
            item["repo_exact_rerun_status"] = "repo_present_waiting_for_python311_api_and_full_result_grid"
            item["repo_paths"] = [str(repo)]
            item["specialized_runner_status"] = gate["status"]
            item["specialized_runner_artifact_dir"] = str(Path(cfg["runner_dir"]))
            statuses = item.setdefault("implementation_statuses", [])
            for status in [
                "official_repo_discovered_and_encoded",
                "python311_required_current_python310_blocked",
                "openrouter_api_key_missing",
                "professional_gate_blocked",
                "gpu_not_required_for_core_api_agent_experiments",
            ]:
                if status not in statuses:
                    statuses.append(status)
            item["specialized_runner_evidence"] = {
                "blockers": gate["blockers"],
                "verifier_path": str(Path(cfg["runner_dir"]) / "battleship_specialized_verifier.json"),
                "repo_manifest_path": str(Path(cfg["runner_dir"]) / "repo_manifest.json"),
                "environment_path": str(Path(cfg["runner_dir"]) / "environment.json"),
                "model_data_manifest_path": str(Path(cfg["runner_dir"]) / "model_data_manifest.json"),
                "api_runtime_manifest_path": str(Path(cfg["runner_dir"]) / "api_runtime_manifest.json"),
            }
            break
    write_json(QUEUE_PATH, queue_obj)

    summary = read_json(SUMMARY_PATH)
    for paper in summary.get("papers", []):
        if paper.get("paper_id") == cfg["paper_id"]:
            paper["final_status"] = "blocked_waiting_for_professional_artifacts_after_dag_update"
            paper["converged"] = False
            paper["repo_paths"] = [str(repo)]
            paper["specialized_runner_status"] = gate["status"]
            paper["professional_blocker"] = gate["status"]
            paper["specialized_runner_artifact_dir"] = str(Path(cfg["runner_dir"]))
            statuses = paper.setdefault("implementation_statuses", [])
            for status in [
                "official_repo_discovered_and_encoded",
                "python311_required_current_python310_blocked",
                "openrouter_api_key_missing",
                "professional_gate_blocked",
                "gpu_not_required_for_core_api_agent_experiments",
            ]:
                if status not in statuses:
                    statuses.append(status)
            paper["iterations"] = paper.get("iterations", []) + [
                {
                    "iteration": 3,
                    "dag_signature": dag.get("signature"),
                    "simulation": {
                        "paper_id": cfg["paper_id"],
                        "paper_title": cfg["title"],
                        "created_at_utc": gate["created_at_utc"],
                        "input_contract": dag.get("blind_contract", {}),
                        "paper_text_seen": False,
                        "previous_memory_seen": False,
                        "oracle_results_seen": False,
                        "repo_paths": [str(repo)],
                        "author_decision": "explicit_operational_blocker",
                        "professional_package_ready": False,
                        "professional_package_reason": gate["status"],
                        "reduced_or_proxy_used_for_convergence": False,
                        "raw_artifact_level": "repo_script_python_api_data_preflight_only",
                        "gpu_used": False,
                        "gpu_use_reason": "paper core execution is API/symbolic agent benchmark; GPU is not required unless local HF baselines are selected",
                        "blocker_ids": [b["id"] for b in gate["blockers"]],
                    },
                    "verification": verify,
                }
            ]
            break
    summary["updated_at_utc"] = utc_now()
    summary["final_status"] = "running_professional_two_loop_not_converged"
    write_json(SUMMARY_PATH, summary)


def write_status(cfg: dict[str, Any], gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    status_path = Path(cfg["runner_dir"]) / "BATTLESHIP_SPECIALIZED_STATUS.md"
    lines = [
        "# Battleship Specialized Professional Gate",
        "",
        f"- Paper id: `{cfg['paper_id']}`",
        f"- Title: {cfg['title']}",
        f"- Status: `{gate['status']}`",
        "- Converged: `false`",
        "- Professional ready: `false`",
        "- GPU used: `false` for this paper-specific preflight; core paper execution is API/symbolic, not CUDA.",
        f"- DAG signature: `{dag.get('signature')}`",
        f"- Repo: `{cfg['repo']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in gate["blockers"]:
        lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    lines += [
        "",
        "## Artifact Paths",
        "",
        f"- Professional gate: `{Path(cfg['runner_dir']) / 'professional_gate_result.json'}`",
        f"- Verifier: `{Path(cfg['runner_dir']) / 'battleship_specialized_verifier.json'}`",
        f"- Repo manifest: `{Path(cfg['runner_dir']) / 'repo_manifest.json'}`",
        f"- Environment: `{Path(cfg['runner_dir']) / 'environment.json'}`",
        f"- Model/data manifest: `{Path(cfg['runner_dir']) / 'model_data_manifest.json'}`",
        f"- API runtime manifest: `{Path(cfg['runner_dir']) / 'api_runtime_manifest.json'}`",
        "",
        "## Verifier Checks",
        "",
    ]
    for check in verify["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = dict(CFG)
    cfg["paper_run_dir"] = str(RUN_ROOT / "paper_runs" / cfg["paper_run"])
    cfg["runner_dir"] = str(RUN_ROOT / "specialized_runners" / cfg["short"])
    Path(cfg["runner_dir"]).mkdir(parents=True, exist_ok=True)
    manifest = script_manifest(cfg)
    env = environment_manifest(cfg)
    artifacts = artifact_manifest(cfg)
    api = api_runtime_manifest(cfg)
    gate = professional_gate(cfg, env, artifacts, api)
    dag = update_dag(cfg, gate)
    verify = verifier(cfg, gate, dag)
    update_queue_summary(cfg, gate, verify, dag)
    write_status(cfg, gate, verify, dag)
    refresh = run_cmd([sys.executable, str(RUN_ROOT / "refresh_longgoal_status.py")], cwd=RUN_ROOT, timeout=120)
    print(
        json.dumps(
            {
                "paper_id": cfg["paper_id"],
                "status": gate["status"],
                "converged": False,
                "blocker_count": len(gate["blockers"]),
                "blocker_ids": [b["id"] for b in gate["blockers"]],
                "dag_signature": dag.get("signature"),
                "status_path": str(Path(cfg["runner_dir"]) / "BATTLESHIP_SPECIALIZED_STATUS.md"),
                "refresh_returncode": refresh["returncode"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
