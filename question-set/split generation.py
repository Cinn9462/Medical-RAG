import pandas as pd
from datasets import load_dataset
import math

# ========== MedMCQA 1000 ==========
print("Loading MedMCQA...")
medmcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
medmcqa_df = pd.DataFrame(medmcqa)

subject_counts = medmcqa_df["subject_name"].value_counts().sort_index()
total = len(medmcqa_df)

targets = {}
fractional = {}
allocated = 0
for subject, count in subject_counts.items():
    exact = count / total * 1000
    targets[subject] = math.floor(exact)
    fractional[subject] = exact - targets[subject]
    allocated += targets[subject]

remaining = 1000 - allocated
for subject in sorted(fractional, key=lambda x: fractional[x], reverse=True)[:remaining]:
    targets[subject] += 1

sampled = []
for subject, target in targets.items():
    subject_df = medmcqa_df[medmcqa_df["subject_name"] == subject]
    sampled.append(subject_df.sample(target))

medmcqa_sampled = pd.concat(sampled).sample(frac=1).reset_index(drop=True)
medmcqa_sampled["source"] = "MEDMCQA"
medmcqa_sampled["id"] = range(1000)

print(f"Total sampled: {len(medmcqa_sampled)}")
print(f"Per-subject counts:\n{medmcqa_sampled['subject_name'].value_counts().sort_index()}\n")

medmcqa_sampled = medmcqa_sampled[["id", "question", "opa", "opb", "opc", "opd", "cop", "subject_name", "source"]]
medmcqa_sampled.to_csv("medmcqa_1000.csv", index=False)

# ========== MMLU 1000 ==========
print("Loading MMLU...")
subjects = [
    "anatomy", "clinical-knowledge", "college-biology", "college-chemistry",
    "college-medicine", "high-school-biology", "human-aging", "medical-genetics",
    "nutrition", "professional-medicine", "professional-psychology", "virology"
]

all_mmlu = []
for s in subjects:
    ds = load_dataset("ekacare/mmlu-medical-mcqs-evaluation-dataset", s, split="test")
    df = pd.DataFrame(ds)
    df["subject"] = s
    all_mmlu.append(df)

mmlu_combined = pd.concat(all_mmlu, ignore_index=True)

subject_counts = mmlu_combined["subject"].value_counts().sort_index()
total = len(mmlu_combined)

targets = {}
fractional = {}
allocated = 0
for subject, count in subject_counts.items():
    exact = count / total * 1000
    targets[subject] = math.floor(exact)
    fractional[subject] = exact - targets[subject]
    allocated += targets[subject]

remaining = 1000 - allocated
for subject in sorted(fractional, key=lambda x: fractional[x], reverse=True)[:remaining]:
    targets[subject] += 1

sampled = []
for subject, target in targets.items():
    subject_df = mmlu_combined[mmlu_combined["subject"] == subject]
    sampled.append(subject_df.sample(target))

mmlu_sampled = pd.concat(sampled).sample(frac=1).reset_index(drop=True)
mmlu_sampled["source"] = "MMLU"
mmlu_sampled["id"] = range(1000)

print(f"Total sampled: {len(mmlu_sampled)}")
print(f"Per-subject counts:\n{mmlu_sampled['subject'].value_counts().sort_index()}\n")

mmlu_sampled = mmlu_sampled[["id", "centerpiece", "options", "correct_options_idx", "subject", "source"]]
mmlu_sampled.to_csv("mmlu_1000.csv", index=False)

# ========== MedMCQA 20 ==========
print("Loading MedMCQA 20...")
medmcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
medmcqa_df = pd.DataFrame(medmcqa)
sample_20 = medmcqa_df.sample(n=20).reset_index(drop=True)
sample_20["source"] = "MEDMCQA"
sample_20["id"] = range(20)
sample_20 = sample_20[["id", "question", "opa", "opb", "opc", "opd", "cop", "subject_name", "source"]]

sample_20.to_csv("medmcqa_20.csv", index=False)
print(f"Saved {len(sample_20)} questions to medmcqa_20.csv")