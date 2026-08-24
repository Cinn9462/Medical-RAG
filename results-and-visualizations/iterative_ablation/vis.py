import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# region Ablation Calibration
thresholds = []
accuracies = []
avg_times = []

for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    mcqadf = pd.read_csv(f"results-and-visualizations/iterative_ablation/res/medmcqa_results_{threshold}.csv")
    qadf = pd.read_csv(f"results-and-visualizations/iterative_ablation/res/medqa_results_{threshold}.csv")
    mcqadf['probE'] = 0.0  # Add a column for probE with default value 0.0
    df = pd.concat([mcqadf, qadf], ignore_index=True)
    df['choice'] = df[["probA", "probB", "probC", "probD", "probE"]].idxmax(axis=1).str.extract(r'prob(\w)')[0]
    df['result'] = df[["probA", "probB", "probC", "probD", "probE"]].values.argmax(axis=1) == df['answer']

    accuracy = (df['result'].sum() / len(df) * 100)
    avg_time = df['time'].mean()
    thresholds.append(threshold)
    accuracies.append(accuracy)
    avg_times.append(avg_time)

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(thresholds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Overall Accuracy (%)', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(58, 66)

ax2 = ax1.twinx()
ax2.plot(thresholds, avg_times, marker='o', linestyle='-', color='r', label='Avg Time')
ax2.set_ylabel('Average Time (s)', fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax1.axvline(x=0.3, color='g', linestyle='--')
ax1.set_xticks(thresholds[::2])
ax1.set_xticklabels([f'{t:.1f}' for t in thresholds[::2]])
ax1.set_xlim(0, 1.0)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Calibration Curve', fontsize=14)
plt.xticks(thresholds, rotation=45)
plt.tight_layout()
plt.savefig('accuracy_vs_threshold_with_time.png', dpi=300)
plt.show()
# endregion Ablation Calibration