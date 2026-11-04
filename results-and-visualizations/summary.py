import pandas as pd

RAG_VISUALIZATIONS = False

# Read both dataframes
mcqa_df = pd.read_csv("results-and-visualizations/baseline/medmcqa_results.csv")   # 4 options (probA‑D)
qa_df   = pd.read_csv("results-and-visualizations/baseline/medqa_results.csv")     # 5 options (probA‑E)

# Add missing probE column to the 4‑option dataframe
mcqa_df['probE'] = 0.0

# Combine into one DataFrame
df = pd.concat([mcqa_df, qa_df], ignore_index=True)
print(df["time"].mean())

# ---- Compute derived columns ----
prob_cols = ['probA', 'probB', 'probC', 'probD', 'probE']
THRESHOLD = [0, 0.40, 0.70, 0.90, 1.01]
labels = ['Low', 'Medium', 'High', 'Very High']

df['confidence'] = pd.cut(df[prob_cols].max(axis=1),
                          bins=THRESHOLD, labels=labels, right=False)
df['choice'] = df[prob_cols].idxmax(axis=1).str.extract(r'prob(\w)')[0]
df['result'] = df[prob_cols].values.argmax(axis=1) == df['answer']

answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
df['answer_letter'] = df['answer'].map(answer_map)
df['p_top1'] = df[prob_cols].max(axis=1)
df['p_ans'] = df[prob_cols].values[range(len(df)), df['answer'].values]
df['margin'] = df['p_top1'] - df['p_ans']

print (sum(df["result"])/len(df["result"]))