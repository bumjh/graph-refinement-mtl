"""Append a compact experiment row from test_threshold_metrics.xlsx.

Output columns:
    Method, seed, AUROC, AUPRC, Sensitivity @95% Spe, Specificity, F1 (optimal_threshold), AUX mean auc

Example:
    python append_experiment_summary.py ^
        --metrics-xlsx checkpoints/conditional_rg_refine/test_threshold_metrics.xlsx ^
        --method ConditionalRG ^
        --seed 123
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "experiment_summary.xlsx"
SUMMARY_COLUMNS = [
    "Method",
    "seed",
    "AUROC↑",
    "AUPRC↑",
    "Sensitivity @95% Spe↑",
    "Specificity↑",
    "F1 (opitmal_threshold) ↑",
    "AUX mean auc",
    "metrics_xlsx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one experiment row from test_threshold_metrics.xlsx.")
    parser.add_argument("--metrics-xlsx", type=str, default=r'C:\Users\bum\PycharmProjects\RGS\checkpoints\ablation_a_rgb\test_threshold_metrics.xlsx', help="Input test_threshold_metrics.xlsx.")
    parser.add_argument("--output-xlsx", type=str, default=str(DEFAULT_OUTPUT), help="Summary workbook to append/update.")
    parser.add_argument("--method", type=str, default='baseline_mtl', help="Method name. Defaults to checkpoint parent folder.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed. Defaults to parsing checkpoint args/path when possible.")
    parser.add_argument("--spec-mode", type=str, default="specificity_0.95", help="Row in summary sheet for Sensitivity @95% Spe.")
    parser.add_argument("--f1-mode", type=str, default="f1_optimal", help="Row in summary sheet for F1.")
    parser.add_argument(
        "--aux-threshold-source",
        type=str,
        default="validation_f1",
        help="Preferred auxiliary_metrics threshold_source. Falls back to fixed_0.5 if absent.",
    )
    parser.add_argument("--replace", action="store_true", help="Replace existing row with same Method and seed.")
    return parser.parse_args()


def latest_metrics_xlsx() -> Path:
    root = Path(__file__).resolve().parent
    candidates = sorted(
        root.rglob("test_threshold_metrics.xlsx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No test_threshold_metrics.xlsx found under {root}")
    return candidates[0]


def first_value(frame: pd.DataFrame, column: str) -> Optional[object]:
    if column not in frame.columns or frame.empty:
        return None
    value = frame[column].iloc[0]
    return None if pd.isna(value) else value


def row_by_mode(summary: pd.DataFrame, mode: str) -> pd.Series:
    if "mode" not in summary.columns:
        raise ValueError("summary sheet must contain a 'mode' column.")
    matched = summary.loc[summary["mode"].astype(str) == mode]
    if matched.empty:
        available = ", ".join(summary["mode"].astype(str).tolist())
        raise ValueError(f"Mode '{mode}' not found in summary sheet. Available modes: {available}")
    return matched.iloc[0]


def infer_method(metrics_path: Path, run_info: pd.DataFrame, method_arg: Optional[str]) -> str:
    if method_arg:
        return method_arg
    checkpoint = first_value(run_info, "checkpoint")
    if checkpoint:
        return Path(str(checkpoint)).parent.name
    return metrics_path.parent.name


def infer_seed(metrics_path: Path, seed_arg: Optional[int]) -> Optional[int]:
    if seed_arg is not None:
        return int(seed_arg)
    match = re.search(r"seed[_-]?(\d+)", str(metrics_path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def aux_mean_auc(auxiliary_metrics: pd.DataFrame, preferred_source: str) -> float:
    if auxiliary_metrics.empty:
        return float("nan")
    frame = auxiliary_metrics.copy()
    if "threshold_source" in frame.columns:
        preferred = frame.loc[frame["threshold_source"].astype(str) == preferred_source]
        if preferred.empty and preferred_source != "fixed_0.5":
            preferred = frame.loc[frame["threshold_source"].astype(str) == "fixed_0.5"]
        if not preferred.empty:
            frame = preferred
    if "auroc" not in frame.columns:
        raise ValueError("auxiliary_metrics sheet must contain an 'auroc' column.")
    return float(pd.to_numeric(frame["auroc"], errors="coerce").dropna().mean())


def build_summary_row(metrics_path: Path, args: argparse.Namespace) -> dict:
    workbook = pd.ExcelFile(metrics_path)
    summary = pd.read_excel(workbook, sheet_name="summary")
    run_info = pd.read_excel(workbook, sheet_name="run_info") if "run_info" in workbook.sheet_names else pd.DataFrame()
    auxiliary_metrics = (
        pd.read_excel(workbook, sheet_name="auxiliary_metrics")
        if "auxiliary_metrics" in workbook.sheet_names
        else pd.DataFrame()
    )

    spec_row = row_by_mode(summary, args.spec_mode)
    f1_row = row_by_mode(summary, args.f1_mode)
    return {
        "Method": infer_method(metrics_path, run_info, args.method),
        "seed": infer_seed(metrics_path, args.seed),
        "AUROC↑": float(spec_row["auroc"]),
        "AUPRC↑": float(spec_row["auprc"]),
        "Sensitivity @95% Spe↑": float(spec_row["sensitivity"]),
        "Specificity↑": float(spec_row["specificity"]),
        "F1 (opitmal_threshold) ↑": float(f1_row["f1_score"]),
        "AUX mean auc": aux_mean_auc(auxiliary_metrics, args.aux_threshold_source),
        "metrics_xlsx": str(metrics_path),
    }


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_xlsx).resolve() if args.metrics_xlsx else latest_metrics_xlsx().resolve()
    output_path = Path(args.output_xlsx).resolve()
    row = build_summary_row(metrics_path, args)

    if output_path.exists():
        table = pd.read_excel(output_path)
    else:
        table = pd.DataFrame()
    new_row = pd.DataFrame([row])
    if args.replace and not table.empty and {"Method", "seed"}.issubset(table.columns):
        same = (table["Method"].astype(str) == str(row["Method"])) & (table["seed"].astype(str) == str(row["seed"]))
        table = table.loc[~same].copy()
    table = pd.concat([table, new_row], ignore_index=True)
    extra_columns = [column for column in table.columns if column not in SUMMARY_COLUMNS]
    table = table.reindex(columns=[*SUMMARY_COLUMNS, *extra_columns])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="summary", index=False)

    print(f"Appended experiment summary -> {output_path}")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
