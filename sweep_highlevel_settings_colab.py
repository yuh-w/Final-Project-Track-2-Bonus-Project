#!/usr/bin/env python3
"""Run several high-level CEM settings and summarize official evaluations.

This is intended for Colab. It trains each named setting with train_mlp_cem.py,
then evaluates the global top training candidates across all settings with
run_track_bonus.py. Results are collected into JSON and CSV files.
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
        "edge_slowdown_margin_norm": 0.15,
        "max_command_delta": 0.15,
        "boundary_safety_margin_m": 0.05,
    },
    {
        "name": "robust_38",
        "command_filter_alpha": 0.35,
        "max_straight_speed_mps": 3.8,
        "max_curve_speed_mps": 3.2,
        "max_lateral_speed_mps": 0.22,
        "max_yaw_rate_radps": 0.60,
        "edge_slowdown_margin_norm": 0.15,
        "max_command_delta": 0.15,
        "boundary_safety_margin_m": 0.05,
    },
    {
        "name": "robust_36",
        "command_filter_alpha": 0.33,
        "max_straight_speed_mps": 3.6,
        "max_curve_speed_mps": 3.0,
        "max_lateral_speed_mps": 0.20,
        "max_yaw_rate_radps": 0.58,
        "edge_slowdown_margin_norm": 0.10,
        "max_command_delta": 0.14,
        "boundary_safety_margin_m": 0.05,
    },
    {
        "name": "smooth_40",
        "command_filter_alpha": 0.30,
        "max_straight_speed_mps": 4.0,
        "max_curve_speed_mps": 3.1,
        "max_lateral_speed_mps": 0.20,
        "max_yaw_rate_radps": 0.58,
        "edge_slowdown_margin_norm": 0.20,
        "max_command_delta": 0.11,
        "boundary_safety_margin_m": 0.05,
    },
    {
        "name": "aggressive_exit",
        "command_filter_alpha": 0.39,
        "max_straight_speed_mps": 5.2,
        "max_curve_speed_mps": 3.5,
        "max_lateral_speed_mps": 0.25,
        "max_yaw_rate_radps": 0.70,
        "edge_slowdown_margin_norm": 0.10,
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
    parser.add_argument("--top-k-results", type=int, default=5, help="Number of generation-best planners to save per training run.")
    parser.add_argument("--eval-top-k", type=int, default=5, help="Number of global top training candidates to evaluate after training.")
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


def collect_global_candidates(
    trained_settings: list[tuple[dict[str, Any], Path]],
    top_k: int,
    output_root: Path,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for setting, train_dir in trained_settings:
        summary_path = train_dir / "top_results" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in summary:
            planner_config = Path(str(record["planner_config"]))
            if not planner_config.is_absolute():
                planner_config = ROOT / planner_config
            if not planner_config.exists():
                continue
            candidate_name = f"top_{int(record.get('rank', 0))}"
            candidates.append(
                {
                    "setting": setting,
                    "candidate_name": candidate_name,
                    "planner_config": planner_config,
                    "training_score": float(record["score"]),
                    "training_iteration": int(record.get("iteration", -1)),
                    "training_candidate": int(record.get("candidate", -1)),
                    "training_lap_completion": float(record.get("lap_completion", 0.0)),
                    "training_finish_time": record.get("finish_time"),
                    "training_mean_progress_speed": float(record.get("mean_progress_speed", 0.0)),
                    "training_fall": bool(record.get("fall", False)),
                    "training_boundary_violation": bool(record.get("boundary_violation", False)),
                }
            )

    candidates.sort(key=lambda row: row["training_score"], reverse=True)
    selected = candidates[: max(1, int(top_k))]
    for global_rank, candidate in enumerate(selected, start=1):
        candidate["global_training_rank"] = global_rank

    summary_payload = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in candidate.items()
            if key != "setting"
        }
        | {"setting": candidate["setting"]["name"]}
        for candidate in selected
    ]
    (output_root / "global_training_top_candidates.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    return selected


def summarize_result(
    *,
    setting: dict[str, Any],
    candidate_name: str,
    eval_seed: int,
    eval_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_path = eval_dir / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    scores = payload["scores"]
    row = {
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
    if extra:
        row.update(extra)
    return row


def write_summaries(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "sweep_summary.json"
    csv_path = output_root / "sweep_summary.csv"
    robust_json_path = output_root / "robust_summary.json"
    robust_csv_path = output_root / "robust_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        robust_rows = build_robust_summary(rows)
        robust_json_path.write_text(json.dumps(robust_rows, indent=2), encoding="utf-8")
        robust_fieldnames = list(robust_rows[0].keys())
        with robust_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=robust_fieldnames)
            writer.writeheader()
            writer.writerows(robust_rows)

    print(f"\nSaved summary JSON: {json_path}", flush=True)
    print(f"Saved summary CSV:  {csv_path}", flush=True)
    if rows:
        print(f"Saved robust JSON:  {robust_json_path}", flush=True)
        print(f"Saved robust CSV:   {robust_csv_path}", flush=True)


def build_robust_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["setting"]), str(row["candidate"])), []).append(row)

    robust_rows: list[dict[str, Any]] = []
    for (setting, candidate), group in grouped.items():
        eval_count = len(group)
        fall_count = sum(1 for row in group if bool(row["fall"]))
        boundary_count = sum(1 for row in group if bool(row["boundary_violation"]))
        finished = [row for row in group if row["finish_time"] is not None]
        composite_values = [float(row["composite_score"]) for row in group]
        lap_values = [float(row["lap_completion"]) for row in group]
        speed_values = [float(row["mean_progress_speed"]) for row in group]
        finish_values = [float(row["finish_time"]) for row in finished]
        success_count = sum(
            1
            for row in group
            if row["finish_time"] is not None and not bool(row["fall"]) and not bool(row["boundary_violation"])
        )

        # Prefer candidates that finish every seed, then faster candidates.
        robust_score = (
            100.0 * (success_count / eval_count)
            + 10.0 * min(lap_values)
            + sum(composite_values) / eval_count
            - 20.0 * (fall_count / eval_count)
            - 10.0 * (boundary_count / eval_count)
        )
        if finish_values:
            robust_score -= 0.1 * (sum(finish_values) / len(finish_values))

        robust_rows.append(
            {
                "setting": setting,
                "candidate": candidate,
                "eval_count": eval_count,
                "success_count": success_count,
                "success_rate": success_count / eval_count,
                "fall_count": fall_count,
                "boundary_count": boundary_count,
                "min_lap_completion": min(lap_values),
                "mean_lap_completion": sum(lap_values) / eval_count,
                "mean_composite_score": sum(composite_values) / eval_count,
                "mean_progress_speed": sum(speed_values) / eval_count,
                "mean_finish_time": None if not finish_values else sum(finish_values) / len(finish_values),
                "robust_score": robust_score,
            }
        )

    robust_rows.sort(
        key=lambda row: (
            row["success_rate"],
            -row["fall_count"],
            row["min_lap_completion"],
            row["robust_score"],
        ),
        reverse=True,
    )
    return robust_rows


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
    trained_settings: list[tuple[dict[str, Any], Path]] = []
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
            "--top-k-results",
            str(args.top_k_results),
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
        trained_settings.append((setting, train_dir))

    if args.skip_eval:
        print("\nTraining finished. Skipping official evaluation because --skip-eval was set.", flush=True)
        write_summaries(args.output_root, rows)
        return

    print("\n=== Training finished. Selecting global top candidates for official evaluation. ===", flush=True)
    global_candidates = collect_global_candidates(trained_settings, args.eval_top_k, args.output_root)
    print(
        f"Officially evaluating global top {len(global_candidates)} training candidates "
        f"across {len(trained_settings)} setting(s).",
        flush=True,
    )
    for candidate in global_candidates:
        setting = candidate["setting"]
        name = setting["name"]
        candidate_name = str(candidate["candidate_name"])
        planner_config = Path(candidate["planner_config"])
        global_rank = int(candidate["global_training_rank"])
        global_candidate_name = f"global_{global_rank:02d}_{name}_{candidate_name}"
        for eval_seed in eval_seeds:
            eval_dir = args.eval_output_root / "global_top" / global_candidate_name / f"seed_{eval_seed}"
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
                f"{global_candidate_name}_seed_{eval_seed}",
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
                    candidate_name=global_candidate_name,
                    eval_seed=eval_seed,
                    eval_dir=eval_dir,
                    extra={
                        "global_training_rank": global_rank,
                        "source_candidate": candidate_name,
                        "training_score": float(candidate["training_score"]),
                        "training_iteration": int(candidate["training_iteration"]),
                        "training_candidate": int(candidate["training_candidate"]),
                        "training_lap_completion": float(candidate["training_lap_completion"]),
                        "training_finish_time": candidate["training_finish_time"],
                        "training_mean_progress_speed": float(candidate["training_mean_progress_speed"]),
                        "training_fall": bool(candidate["training_fall"]),
                        "training_boundary_violation": bool(candidate["training_boundary_violation"]),
                    },
                )
            )
            write_summaries(args.output_root, rows)

    write_summaries(args.output_root, rows)


if __name__ == "__main__":
    main()
