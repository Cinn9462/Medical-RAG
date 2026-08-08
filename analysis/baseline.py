import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/baseline.csv")

prob_cols = ['probA', 'probB', 'probC', 'probD']
THRESHOLD = [0, 0.40, 0.70, 0.90, 1.01]
labels = ['Low', 'Medium', 'High', 'Very High']

df['confidence'] = pd.cut(df[prob_cols].max(axis=1), bins=THRESHOLD, labels=labels, right=False)
df['choice'] = df[prob_cols].idxmax(axis=1).str.extract(r'prob(\w)')[0]
df['result'] = df[prob_cols].values.argmax(axis=1) == df['answer']
answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
df['answer_letter'] = df['answer'].map(answer_map)
df['p_top1'] = df[prob_cols].max(axis=1)
df['p_ans'] = df[prob_cols].values[range(len(df)), df['answer'].values]
df['margin'] = df['p_top1'] - df['p_ans']

medmcqa_df = df[df['source'].str.lower() == 'medmcqa']
mmlu_df = df[df['source'].str.lower() == 'mmlu']

# Pie Charts
data_list = [
    (df['answer_letter'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Answers (Overall)"),
    (medmcqa_df['answer_letter'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Answers (MEDMCQA)"),
    (mmlu_df['answer_letter'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Answers (MMLU)"),
    (df['choice'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Predictions (Overall)"),
    (medmcqa_df['choice'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Predictions (MEDMCQA)"),
    (mmlu_df['choice'].value_counts().reindex(['A','B','C','D'], fill_value=0), "Predictions (MMLU)"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

for ax, (counts, title) in zip(axes, data_list):
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors)
    ax.set_title(f"{title} (n={sum(counts)})")
    ax.axis('equal')

plt.tight_layout()
plt.savefig('answer_pie_charts.png', bbox_inches='tight')

# Accuracy by Subject
medmcqa_acc = medmcqa_df.groupby('subject')['result'].mean().reset_index()
medmcqa_avg = medmcqa_df['result'].mean()

mmlu_acc = mmlu_df.groupby('subject')['result'].mean().reset_index()
mmlu_avg = mmlu_df['result'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

ax1.barh(medmcqa_acc['subject'], medmcqa_acc['result'], color='steelblue', alpha=0.8)
ax1.axvline(x=medmcqa_avg, color='red', linestyle='--', label=f'Average = {medmcqa_avg:.3f}')
ax1.set_title('MEDMCQA'); ax1.set_xlabel('Accuracy'); ax1.set_ylabel('Subject')
ax1.set_xlim(0, 1); ax1.legend()

ax2.barh(mmlu_acc['subject'], mmlu_acc['result'], color='darkorange', alpha=0.8)
ax2.axvline(x=mmlu_avg, color='red', linestyle='--', label=f'Average = {mmlu_avg:.3f}')
ax2.set_title('MMLU'); ax2.set_xlabel('Accuracy'); ax2.set_ylabel('Subject')
ax2.set_xlim(0, 1); ax2.legend()

plt.tight_layout()
plt.savefig('accuracy_by_subject.png', dpi=300)

# Error by confidence
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = medmcqa_df[medmcqa_df['result'] == outcome]['p_top1']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Predicted Probability'); ax.set_ylabel('Density')
ax.set_title('MEDMCQA - Confidence Distribution'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
for outcome, color, label in zip([True, False], ['blue', 'red'], ['Correct', 'Incorrect']):
    subset = mmlu_df[mmlu_df['result'] == outcome]['p_top1']
    if not subset.empty:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
ax.set_xlabel('Maximum Predicted Probability'); ax.set_ylabel('Density')
ax.set_title('MMLU - Confidence Distribution'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
crosstab = pd.crosstab(df['confidence'], df['result'], normalize='index') * 100
crosstab = crosstab.rename(columns={True: 'Correct', False: 'Incorrect'})
crosstab[['Incorrect', 'Correct']].plot(kind='bar', stacked=True, ax=ax, color=['red', 'blue'], edgecolor='black')
ax.set_xlabel('Confidence Level'); ax.set_ylabel('Percentage (%)')
ax.set_title('Outcome by Confidence Level'); ax.legend(title='Outcome')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7); ax.grid(axis='y', alpha=0.3)

for i, conf in enumerate(crosstab.index):
    total = len(df[df['confidence'] == conf])
    ax.text(i, 102, f'n={total}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')

# Probability Density of Correct Answer
fig, ax = plt.subplots(figsize=(10, 6))

for outcome, color, label in [(True, '#2ecc71', 'Correct'), (False, '#e74c3c', 'Incorrect')]:
    subset = df[df['result'] == outcome]['margin']
    if not subset.empty and subset.nunique() > 1:
        sns.kdeplot(subset, ax=ax, color=color, label=label, linewidth=2.5, fill=True, alpha=0.25)

ax.set_xlabel('p(top1) − p(answer)')
ax.set_ylabel('Relative Density')
ax.set_title('Relative Density of p(top1) − p(answer)')
ax.set_xlim(0, 1)
ax.legend(title='Prediction')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('margin_density.png', dpi=300, bbox_inches='tight')

# Time
# print(f"Average time taken for each question: {df['time'].mean() :.2f} ± {df['time'].std() :.2f} seconds".format())
# Average time taken for each question: 0.58 ± 1.41 seconds