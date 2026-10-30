"""
Combine "never run" and "always run" results per dataset using a confidence margin.

For each question:
  margin = top-1 prob - top-2 prob, computed from the NEVER-run (first-pass) probabilities
  margin >  THRESH -> keep the never-run row   (confident, no second retrieval)
  margin <= THRESH -> keep the always-run row  (not confident, use second retrieval)

Questions whose ids appear in the EXCLUDE_IDS csv for that dataset are dropped
from both files before combining (e.g. exclude the calibration ids to evaluate
on the held-out test set only).
"""
import os
import numpy as np
import pandas as pd

# ----------------------------- CONFIG -----------------------------
FILES = {
    "medmcqa": {
        "never":  "results-and-visualizations/rag_baseline/medmcqa_results.csv",
        "always": "results-and-visualizations/hyde_style_rag/medmcqa_results.csv",
    },
    "medqa": {
        "never":  "results-and-visualizations/rag_baseline/medqa_results.csv",
        "always": "results-and-visualizations/hyde_style_rag/medqa_results.csv",
    },
}

# CSV of ids to exclude per dataset (must have an "id" column).
# Set a dataset's value to None to exclude nothing for it.
EXCLUDE_IDS = {
    "medmcqa": "results-and-visualizations\\iterative_ablation\\res_split\\medmcqa_calibration_ids.csv",
    "medqa":   "results-and-visualizations\\\\iterative_ablation\\res_split\\medqa_calibration_ids.csv",
}
ID_COL = "id"

OUT_DIR = "res_combined"

PROB_COLS = ["probA", "probB", "probC", "probD", "probE"]
THRESH = 0.4                                   # threshold for the saved combined csvs
SWEEP_VALUES = [round(x * 0.1, 1) for x in range(0, 11)]
# ------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def load_exclude_ids(prefix):
    path = EXCLUDE_IDS.get(prefix)
    if not path:
        return set()
    return set(pd.read_csv(path)[ID_COL])


def load_pair(prefix):
    """Load both files, drop excluded ids, keep shared ids, align row order."""
    a = pd.read_csv(FILES[prefix]["never"])
    b = pd.read_csv(FILES[prefix]["always"])

    exclude = load_exclude_ids(prefix)
    n_before = len(a)
    a = a[~a["id"].isin(exclude)]
    b = b[~b["id"].isin(exclude)]
    print(f"{prefix}: excluded {n_before - len(a)} of {n_before} rows "
          f"({len(exclude)} ids listed)")

    common = sorted(set(a["id"]) & set(b["id"]))
    a = a[a["id"].isin(common)].sort_values("id").reset_index(drop=True)
    b = b[b["id"].isin(common)].sort_values("id").reset_index(drop=True)
    return a, b


def combine(a, b, thresh):
    cols = [c for c in PROB_COLS if c in a.columns and a[c].notna().any()]
    sorted_p = np.sort(a[cols].to_numpy(), axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    use_never = margin > thresh

    out = pd.concat([a[use_never], b[~use_never]]).sort_values("id").reset_index(drop=True)
    out["margin"] = out["id"].map(dict(zip(a["id"], margin)))
    out["chosen_from"] = out["id"].map(dict(zip(a["id"], np.where(use_never, "never", "always"))))
    return out


def summarize(df):
    """Returns (n, correct, time_sum), excluding errored rows."""
    df = df[df["error"] == False]
    cols = [c for c in PROB_COLS if c in df.columns and df[c].notna().any()]
    pred = np.argmax(df[cols].to_numpy(), axis=1)
    correct = pred == df["answer"].astype(int).to_numpy()
    return len(df), int(correct.sum()), float(df["time"].sum())


def run_threshold(pairs, thresh, save=False):
    row = {"thresh": thresh}
    tot_n = tot_c = tot_t = 0
    for prefix, (a, b) in pairs.items():
        out = combine(a, b, thresh)
        n, c, t = summarize(out)
        tot_n, tot_c, tot_t = tot_n + n, tot_c + c, tot_t + t
        row[f"{prefix} acc (%)"] = 100 * c / n
        row[f"{prefix} time (s)"] = t / n
        row[f"{prefix} always-run (%)"] = 100 * (out["chosen_from"] == "always").mean()
        if save:
            out.to_csv(os.path.join(OUT_DIR, f"{prefix}_combined_{thresh}.csv"), index=False)
    row["overall acc (%)"] = 100 * tot_c / tot_n
    row["overall time (s)"] = tot_t / tot_n
    row["n"] = tot_n
    return row


if __name__ == "__main__":
    pairs = {p: load_pair(p) for p in FILES}

    print(f"\nThreshold {THRESH}")
    print(pd.DataFrame([run_threshold(pairs, THRESH, save=True)]).round(3).to_string(index=False))

    sweep = pd.DataFrame([run_threshold(pairs, t) for t in SWEEP_VALUES]).round(3)
    sweep.to_csv(os.path.join(OUT_DIR, "sweep.csv"), index=False)
    print("\nSweep:")
    print(sweep.to_string(index=False))