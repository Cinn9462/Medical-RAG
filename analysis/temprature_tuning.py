"""
5
Brier Score: 0.179033
ROC AUC:     0.835165
7.5
Brier Score: 0.153808
ROC AUC:     0.846154
8
Brier Score: 0.149773
ROC AUC:     0.846154
"""

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

df = pd.read_csv("results/temprature_tuning/7.5.csv")

df['result'] = df[['probA','probB','probC','probD']].values.argmax(axis=1) == df['answer']
probs = df[['probA','probB','probC','probD']].values
pred_class = probs.argmax(axis=1)
confidence = probs[np.arange(len(probs)), pred_class]

y_true = df["result"].astype(int)

brier = brier_score_loss(y_true, confidence)
roc_auc = roc_auc_score(y_true, confidence)

print(f"Brier Score: {brier:.6f}")
print(f"ROC AUC:     {roc_auc:.6f}")