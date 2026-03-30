import numpy as np
import pandas as pd

# ================================
# STOCKS
# ================================
stocks = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN',
          'NVDA', 'NFLX', 'IBM', 'ORCL', 'JPM']

# ================================
# METRIC FORMULAS
# ================================
# Accuracy  = (TP + TN) / Total
# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)
# F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)

print("=" * 65)
print("         STOCK MARKET ANN — EVALUATION METRICS")
print("=" * 65)
print()
print("Metric Formulas Used:")
print("  Accuracy  = (TP + TN) / Total")
print("  Precision = TP / (TP + FP)")
print("  Recall    = TP / (TP + FN)")
print("  F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)")
print()

# ================================
# PER-STOCK CALCULATION
# ================================
rows = []

for stock in stocks:
    cm = np.load(f"metrics/{stock}_cm.npy")

    # Confusion matrix layout:
    # [[TN, FP],
    #  [FN, TP]]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    total = TP + TN + FP + FN

    accuracy  = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    rows.append({
        'Stock':     stock,
        'TP':        int(TP),
        'TN':        int(TN),
        'FP':        int(FP),
        'FN':        int(FN),
        'Accuracy':  round(accuracy,  4),
        'Precision': round(precision, 4),
        'Recall':    round(recall,    4),
        'F1':        round(f1,        4),
    })

df = pd.DataFrame(rows)

# ================================
# PRINT CONFUSION MATRIX VALUES
# ================================
print("-" * 65)
print(f"{'Stock':<8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
print("-" * 65)
for _, r in df.iterrows():
    print(f"{r['Stock']:<8} {r['TP']:>6} {r['TN']:>6} {r['FP']:>6} {r['FN']:>6}")
print()

# ================================
# PRINT METRICS TABLE
# ================================
print("-" * 65)
print(f"{'Stock':<8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)
for _, r in df.iterrows():
    print(f"{r['Stock']:<8} "
          f"{r['Accuracy']*100:>9.2f}% "
          f"{r['Precision']*100:>9.2f}% "
          f"{r['Recall']*100:>9.2f}% "
          f"{r['F1']*100:>9.2f}%")

print("-" * 65)

# ================================
# AVERAGES
# ================================
avg_acc  = df['Accuracy'].mean()
avg_prec = df['Precision'].mean()
avg_rec  = df['Recall'].mean()
avg_f1   = df['F1'].mean()

print(f"{'AVERAGE':<8} "
      f"{avg_acc*100:>9.2f}% "
      f"{avg_prec*100:>9.2f}% "
      f"{avg_rec*100:>9.2f}% "
      f"{avg_f1*100:>9.2f}%")
print("=" * 65)
print()

# ================================
# SUMMARY
# ================================
best  = df.loc[df['Accuracy'].idxmax()]
worst = df.loc[df['Accuracy'].idxmin()]

print("SUMMARY")
print("-" * 40)
print(f"  Avg Accuracy  : {avg_acc*100:.2f}%")
print(f"  Avg Precision : {avg_prec*100:.2f}%")
print(f"  Avg Recall    : {avg_rec*100:.2f}%")
print(f"  Avg F1-Score  : {avg_f1*100:.2f}%")
print(f"  Best Stock    : {best['Stock']} ({best['Accuracy']*100:.2f}%)")
print(f"  Worst Stock   : {worst['Stock']} ({worst['Accuracy']*100:.2f}%)")
print()
