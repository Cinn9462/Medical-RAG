"""
Compare a baseline (no-iteration) results file against a dynamic (iterating) results file.

For each dataset:
  - questions where the DYNAMIC file has confident >= ITER_MIN are "iterated"
  - those ids are recorded and flagged in the baseline file (new column: iterated)
  - accuracy / time are reported for iterated vs non-iterated vs all questions,
    for both files, plus iterated / total
"""
import os
import numpy as np
import pandas as pd

# ----------------------------- CONFIG -----------------------------
FILES = {
    "medmcqa": {
        "baseline": "results-and-visualizations/rag_rr_baseline/medmcqa_results.csv",
        "dynamic":  "results-and-visualizations/iterative_0.4/medmcqa_results.csv",
    },
    "medqa": {
        "baseline": "results-and-visualizations/rag_rr_baseline/medqa_results.csv",
        "dynamic":  "results-and-visualizations/iterative_0.4/medqa_results.csv",
    },
}

# CSV of ids to exclude per dataset (must have an ID_COL column).
# Set a dataset's value to None to exclude nothing for it.
EXCLUDE_IDS = {
    "medmcqa": "results-and-visualizations\\iterative_ablation\\res_split\\medmcqa_calibration_ids.csv",
    "medqa":   "results-and-visualizations\\iterative_ablation\\res_split\\medmcqa_calibration_ids.csv",
}
ID_COL = "id"

OUT_DIR = "iteration_compare"
PROB_COLS = ["probA", "probB", "probC", "probD", "probE"]
ITER_MIN = 2          # confident >= ITER_MIN means the dynamic run iterated
# ------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def add_correct(df):
    cols = [c for c in PROB_COLS if c in df.columns and df[c].notna().any()]
    df = df.copy()
    df["pred"] = np.argmax(df[cols].to_numpy(), axis=1)
    df["correct"] = df["pred"] == df["answer"].astype(int).to_numpy()
    return df


def load_exclude_ids(prefix):
    path = EXCLUDE_IDS.get(prefix)
    if not path:
        return set()
    return set(pd.read_csv(path)[ID_COL])


def load_merged(prefix):
    base = pd.read_csv(FILES[prefix]["baseline"])
    dyn = pd.read_csv(FILES[prefix]["dynamic"])

    exclude = load_exclude_ids(prefix)
    n_before = len(base)
    base = base[~base["id"].isin(exclude)]
    dyn = dyn[~dyn["id"].isin(exclude)]
    print(f"{prefix}: excluded {n_before - len(base)} of {n_before} rows "
          f"({len(exclude)} ids listed)")

    base, dyn = add_correct(base), add_correct(dyn)
    keep = ["id", "answer", "pred", "correct", "time", "error"]
    m = base[keep].merge(
        dyn[keep + ["confident"]], on="id", suffixes=("_base", "_dyn"), how="inner"
    )
    m["iterated"] = m["confident"] >= ITER_MIN
    return base, m


def summarize(m, label):
    """Rows for iterated / not iterated / all, using questions with no error in either run."""
    valid = m[(m["error_base"] == False) & (m["error_dyn"] == False)]
    rows = []
    for name, sub in [("iterated", valid[valid["iterated"]]),
                      ("not iterated", valid[~valid["iterated"]]),
                      ("all", valid)]:
        n = len(sub)
        rows.append({
            "dataset": label,
            "subset": name,
            "n": n,
            "baseline acc (%)": 100 * sub["correct_base"].mean() if n else np.nan,
            "dynamic acc (%)": 100 * sub["correct_dyn"].mean() if n else np.nan,
            "baseline time (s)": sub["time_base"].mean() if n else np.nan,
            "dynamic time (s)": sub["time_dyn"].mean() if n else np.nan,
        })
    return rows


if __name__ == "__main__":
    all_merged, rows = [], []
    for prefix in FILES:
        base, m = load_merged(prefix)

        # Record iterated ids, and flag them in the baseline file
        iter_ids = m.loc[m["iterated"], "id"]
        iter_ids.to_csv(os.path.join(OUT_DIR, f"{prefix}_iterated_ids.csv"), index=False)
        flagged = base.drop(columns=["pred", "correct"]).copy()
        flagged["iterated"] = flagged["id"].isin(set(iter_ids))
        flagged.to_csv(os.path.join(OUT_DIR, f"{prefix}_baseline_flagged.csv"), index=False)
        m.to_csv(os.path.join(OUT_DIR, f"{prefix}_comparison.csv"), index=False)

        print(f"{prefix}: iterated {int(m['iterated'].sum())} / {len(m)} "
              f"({100 * m['iterated'].mean():.2f}%)")
        rows += summarize(m, prefix)
        all_merged.append(m)

    total = pd.concat(all_merged, ignore_index=True)
    print(f"overall: iterated {int(total['iterated'].sum())} / {len(total)} "
          f"({100 * total['iterated'].mean():.2f}%)")
    rows += summarize(total, "overall")

    table = pd.DataFrame(rows).round(3)
    table.to_csv(os.path.join(OUT_DIR, "iteration_summary.csv"), index=False)
    print()
    print(table.to_string(index=False))