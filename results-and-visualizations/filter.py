#!/usr/bin/env python3
"""
Aggregate per-question results across configuration folders.

Expected per-question schema (csv): id, source, probA..probE, answer (0-indexed),
time, error, [confident]. Prediction = argmax of prob columns, restricted to the
valid options for the dataset (4 for MedMCQA, 5 for MedQA).

For each folder: load csv files, drop questions whose id is in the exclusion CSV,
then report accuracy (errors excluded), mean latency, and error rate % per dataset
and pooled across both.

Usage: edit the CONFIG block below, then run:  python evaluate_results.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================== CONFIG ===============================
# Command-line flags, if given, override these.

# Parent directory that contains the result folders below. Use "." if the
# folders sit next to this script. Absolute paths also work.
ROOT_DIR = "results-and-visualizations"

# Folders to evaluate. Each entry is relative to ROOT_DIR, or an absolute path.
FOLDERS = [
    "baseline",
    "baseline_cot",
    "hyde_gated",
    "hyde_style_rag",
    "iterative_0.4",
    "iterative_1.0",
    "rag_baseline",
    "rag_cot",
    "rag_rr_baseline",
]

# CSV listing the question ids to exclude (e.g. the calibration set).
EXCLUDED_CSV = "results-and-visualizations\\iterative_ablation\\res_split\\medmcqa_calibration_ids.csv"

# Name of the id column in EXCLUDED_CSV. None = auto-detect (id / question_id / qid,
# else first column). If the CSV also has a "source" column, matching is done on
# (source, id), which avoids MedQA/MedMCQA id collisions.
EXCLUDE_ID_COL = None

# Where to save the summary table. None = print only.
OUT_CSV = "summary.csv"
# ======================================================================

N_OPTIONS = {"medmcqa": 4, "medqa": 5}
PROB_COLS = ["probA", "probB", "probC", "probD", "probE"]


def norm_dataset(text):
    t = str(text).lower().replace("-", "").replace("_", "").replace(" ", "")
    if "medmcqa" in t:  # check first; "medqa" is not a substring of "medmcqa"
        return "medmcqa"
    if "medqa" in t:
        return "medqa"
    return None


def norm_id(x):
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x).strip()


def norm_answer(x):
    """0-indexed int from either an int-like or a letter."""
    try:
        return int(float(x))
    except (ValueError, TypeError):
        s = str(x).strip().upper()
        return ord(s[0]) - 65 if s and s[0].isalpha() else -1


def to_bool(series):
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    ).fillna(True)  # unparseable -> treat as error


def load_file(path):
    df = pd.read_csv(path)
    need = {"id", "source", "answer", "time", "error"} | {"probA", "probB", "probC", "probD"}
    missing = need - set(df.columns)
    if missing:
        print(f"  [skip] {path.name}: missing columns {sorted(missing)}", file=sys.stderr)
        return None

    out = pd.DataFrame({
        "id": df["id"].map(norm_id),
        "dataset": df["source"].map(norm_dataset),
        "latency": pd.to_numeric(df["time"], errors="coerce"),
        "file": path.name,
    })
    if "confident" in df.columns:
        out["confident"] = to_bool(df["confident"])

    probs = df.reindex(columns=PROB_COLS).apply(pd.to_numeric, errors="coerce")
    # mask options that don't exist for the dataset (e.g. E for MedMCQA)
    n_opt = out["dataset"].map(N_OPTIONS).fillna(5).astype(int).values
    mask = np.arange(5)[None, :] >= n_opt[:, None]
    probs = probs.mask(mask)
    no_probs = probs.isna().all(axis=1)
    pred = probs.fillna(-np.inf).values.argmax(axis=1)

    gold = df["answer"].map(norm_answer).values
    out["error"] = to_bool(df["error"]).values | no_probs.values
    out["correct"] = (pred == gold) & ~out["error"].values
    return out.dropna(subset=["dataset"])


def load_folder(folder):
    frames = [f for f in (load_file(p) for p in sorted(folder.rglob("*.csv"))) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_excluded(path, id_col=None):
    ex = pd.read_csv(path)
    lower = {c.lower(): c for c in ex.columns}
    col = id_col or next((lower[c] for c in ("id", "question_id", "qid") if c in lower), ex.columns[0])
    ds_col = next((lower[c] for c in ("source", "dataset", "benchmark") if c in lower), None)
    ids = ex[col].map(norm_id)
    if ds_col:
        return set(zip(ex[ds_col].map(norm_dataset), ids)), None
    return None, set(ids)


def summarize(df, label):
    rows = []
    for name, g in [(n, g) for n, g in df.groupby("dataset")] + [("overall", df)]:
        n, n_err = len(g), int(g["error"].sum())
        valid = g[~g["error"]]
        row = {
            "config": label, "dataset": name, "n_questions": n, "n_errors": n_err,
            "accuracy_%": 100 * valid["correct"].mean() if len(valid) else np.nan,
            "avg_latency_s": g["latency"].mean(),
            "error_rate_%": 100 * n_err / n if n else np.nan,
        }
        if "confident" in g.columns:
            row["pct_confident"] = 100 * g["confident"].mean()
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DIR, help="parent directory containing the folders")
    ap.add_argument("--folders", nargs="*", default=FOLDERS)
    ap.add_argument("--excluded", default=EXCLUDED_CSV, help="CSV of question ids to exclude")
    ap.add_argument("--exclude-id-col", default=EXCLUDE_ID_COL)
    ap.add_argument("--out", default=OUT_CSV, help="path to save summary CSV")
    args = ap.parse_args()

    keyed, id_only = load_excluded(args.excluded, args.exclude_id_col)
    print(f"Loaded {len(keyed) if keyed is not None else len(id_only)} excluded ids "
          f"({'matched on source+id' if keyed is not None else 'matched on id only'})\n")

    all_rows = []
    for name in args.folders:
        folder = Path(args.root) / name  # absolute `name` overrides root automatically
        if not folder.is_dir():
            print(f"[missing] {folder}", file=sys.stderr)
            continue
        print(f"Processing {name} ...")
        df = load_folder(folder)
        if df.empty:
            print("  no usable result files found", file=sys.stderr)
            continue

        before = len(df)
        if keyed is not None:
            drop = np.array([k in keyed for k in zip(df["dataset"], df["id"])])
        else:
            drop = df["id"].isin(id_only).values
        df = df[~drop]
        dups = df.duplicated(subset=["dataset", "id"]).sum()
        if dups:
            print(f"  warning: {dups} duplicate (dataset, id) rows; keeping last", file=sys.stderr)
            df = df.drop_duplicates(subset=["dataset", "id"], keep="last")
        print(f"  {before} rows -> {len(df)} after exclusion")
        all_rows += summarize(df, name)

    if not all_rows:
        sys.exit("No results to report.")

    res = pd.DataFrame(all_rows)
    pd.set_option("display.width", 200)
    print("\n" + res.round(3).to_string(index=False))
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()