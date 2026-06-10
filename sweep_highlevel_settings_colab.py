#!/usr/bin/env python3
"""Run several high-level CEM settings and summarize official evaluations.

This is intended for Colab. It trains each named setting with train_mlp_cem.py,
then evaluates the saved best planner plus the archived top-3 generation
planners with run_track_bonus.py. Results are collected into JSON and CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


SETTINGS: list[dict[str, Any]] = [
    {
        "name": "balanced_fast",
        "command_filter_alpha": 0.37,
        "max_straight_speed_mps": 5.0,
        "max_curve_speed_mps": 3.0,
        "max_lateral_speed_mps": 0.25,
        "max_yaw_rate_radps": 0.65,
        "edge_slowdown_margin_norm": 0.45,
        "max_command_delta": 0.15,
        "boundary_safety_margin_m": 0.05,
    },
    {
        "name": "curve_push",
        "command_filter_alpha": 0.36,
        "max_straight_speed_mps": 5.0,
        "max_curve_speed_mps": 3.3,
        "max_lateral_speed_mps": 0.25,
        "max_yaw_rate_radps": 0.68,
        "edge_slowdown_margin_norm": 0.45,
        "max_command_delta": 0.15,
        "boundary_safety_margin_m": 0.06,
    },
    {
        "name": "late_brake_safe",
        "command_filter_alpha": 0.35,
        "max_straight_speed_mps": 5.2,
        "max_curve_speed_mps": 2.9,
        "max_lateral_speed_mps": 0.25,
        "max_yaw_rate_radps": 0.62,
        "edge_slowdown_margin_norm": 0.50,
        "max_command_delta": 0.14,
        "boundary_safety_margin_m": 0.08,
    },
    {
        "name": "aggressive_exit",
        "command_filter_alpha": 0.39,
        "max_straight_speed_mps": 5.4,
        "max_curve_speed_mps": 3.1,
        "max_lateral_speed_mps": 0.25,
        "max_yaw_rate_radps": 0.70,
        "edge_slowdown_margin_norm": 0.40,
        "max_command_delta": 0.16,
        "boundary_safety_margin_m": 0.05,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "highlevel_sweeps")
    parser.add_argument("--eval-output-root", type=Path, default=ROOT / "track_eval_sweeps")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--population", type=int, default=2048)
    parser.add_argument("--elite-frac", type=float, default=0.20)
    parser.add_argument("--train-eval-seconds", type=float, default=78.0)
    parser.add_argument("--official-eval-seconds", type=float, default=79.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--teacher-steps", type=int, default=1200)
    parser.add_argument("--teacher-samples", type=int, default=8192)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--teacher-lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="20260527",
        help="Comma-separated official-eval seeds, for example: 20260527,20260528,20260529",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="all",
        help="Comma-separated setting names from the SETTINGS list, or 'all'.",
    )
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render evaluation videos. Slower and larger output.")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def selected_settings(setting_names: str) -> list[dict[str, Any]]:
    if setting_names.strip().lower() == "all":
        return SETTINGS
    requested = {name.strip() for name in setting_names.split(",") if name.strip()}
    by_name = {setting["name"]: setting for setting in SETTINGS}
    missing = sorted(requested - set(by_name))
    if missing:
        raise ValueError(f"Unknown setting name(s): {', '.join(missing)}")
    return [setting for setting in SETTINGS if setting["name"] in requested]


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--eval-seeds must contain at least one seed.")
    return seeds


def planner_configs_for(train_dir: Path) -> list[tuple[str, Path]]:
    configs: list[tuple[str, Path]] = []
    best_config = train_dir / "planner_config.json"
    if best_config.exists():
        configs.append(("best_global", best_config))
    top_dir = train_dir / "top_results"
    for rank in range(1, 4):
        top_config = top_dir / f"top_{rank}_planner_config.json"
        if top_config.exists():
            configs.append((f"top_{rank}", top_config))
    return configs


def summarize_result(
    *,
    setting: dict[str, Any],
    candidate_name: str,
    eval_seed: int,
    eval_dir: Path,
) -> dict[str, Any]:
    result_path = eval_dir / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    scores = payload["scores"]
    return {
        "setting": setting["name"],
        "candidate": candidate_name,
        "eval_seed": eval_seed,
        "composite_score": scores.get("composite_score"),
        "lap_completion": metrics.get("lap_completion"),
        "finish_time": metrics.get("finish_time"),
        "mean_progress_speed": metrics.get("mean_progress_speed"),
        "fall": metrics.get("fall"),
        "boundary_violation": metrics.get("boundary_violation"),
        "rms_lateral_error": metrics.get("rms_lateral_error"),
        "max_lateral_error": metrics.get("max_lateral_error"),
        "min_boundary_margin_m": metrics.get("min_boundary_margin_m"),
        "energy_proxy": metrics.get("energy_proxy"),
        "foot_slip_proxy": metrics.get("foot_slip_proxy"),
        "planner_config": payload.get("planner_config"),
        "eval_dir": str(eval_dir),
        **{key: setting[key] for key in setting if key != "name"},
    }


def write_summaries(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "sweep_summary.json"
    csv_path = output_root / "sweep_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved summary JSON: {json_path}", flush=True)
    print(f"Saved summary CSV:  {csv_path}", flush=True)


def main() -> None:
    args = parse_args()
    settings = selected_settings(args.settings)
    eval_seeds = parse_seed_list(args.eval_seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.eval_output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "settings": settings,
        "eval_seeds": eval_seeds,
    }
    (args.output_root / "sweep_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(settings):
        name = setting["name"]
        train_dir = args.output_root / name
        print(f"\n=== Training setting: {name} ===", flush=True)

        train_cmd = [
            sys.executable,
            "train_mlp_cem.py",
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--config",
            str(args.config),
            "--output-dir",
            str(train_dir),
            "--iterations",
            str(args.iterations),
            "--population",
            str(args.population),
            "--elite-frac",
            str(args.elite_frac),
            "--eval-seconds",
            str(args.train_eval_seconds),
            "--hidden-dim",
            str(args.hidden_dim),
            "--teacher-steps",
            str(args.teacher_steps),
            "--teacher-samples",
            str(args.teacher_samples),
            "--teacher-batch-size",
            str(args.teacher_batch_size),
            "--teacher-lr",
            str(args.teacher_lr),
            "--seed",
            str(args.seed + setting_index),
        ]
        if args.force_cpu:
            train_cmd.append("--force-cpu")

        for key, value in setting.items():
            if key == "name":
                continue
            train_cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        run_command(train_cmd)

        if args.skip_eval:
            continue

        for candidate_name, planner_config in planner_configs_for(train_dir):
            for eval_seed in eval_seeds:
                eval_dir = args.eval_output_root / name / candidate_name / f"seed_{eval_seed}"
                eval_cmd = [
                    sys.executable,
                    "run_track_bonus.py",
                    "--checkpoint-dir",
                    str(args.checkpoint_dir),
                    "--planner-config",
                    str(planner_config),
                    "--config",
                    str(args.config),
                    "--output-dir",
                    str(eval_dir),
                    "--entry-name",
                    f"{name}_{candidate_name}_seed_{eval_seed}",
                    "--duration-seconds",
                    str(args.official_eval_seconds),
                    "--seed",
                    str(eval_seed),
                ]
                if args.force_cpu:
                    eval_cmd.append("--force-cpu")
                if not args.render:
                    eval_cmd.append("--no-render")

                run_command(eval_cmd)
                rows.append(
                    summarize_result(
                        setting=setting,
                        candidate_name=candidate_name,
                        eval_seed=eval_seed,
                        eval_dir=eval_dir,
                    )
                )
                write_summaries(args.output_root, rows)

    write_summaries(args.output_root, rows)


if __name__ == "__main__":
    main()
