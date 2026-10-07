import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Load Dataset
df = pd.read_csv('customer-churn-training.csv')

# Non-ML Baseline Rule: High last_login_days and high support_tickets
def non_ml_baseline(row):
    if row['last_login_days'] > 10 and row['support_tickets'] >= 3:
        return 1
    return 0

df['baseline_pred'] = df.apply(non_ml_baseline, axis=1)

print("--- Baseline Heuristic Performance ---")
print(classification_report(df['churned'], df['baseline_pred']))