"""Train simple classifiers on the hourly ML dataset and save metrics/plots."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


data_dir = Path("DATA")
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# Load ML dataset
ml_path = data_dir / "final_ml_dataset.csv"
df = pd.read_csv(ml_path)

# Ensure timestamp parsed (not used for features, but may help diagnostics)
df["startTime"] = pd.to_datetime(df["startTime"], utc=True)

# Features and target
feature_cols = ["consumption_forecast", "wind_forecast", "hour", "day_of_week", "is_weekend"]
X = df[feature_cols]
y = df["is_expensive"]

# Time-based split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scale for logistic regression
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(X_train)
X_test_lr = scaler.transform(X_test)

# Models
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_lr, y_train)

tree = DecisionTreeClassifier(max_depth=5, random_state=0, class_weight="balanced")
tree.fit(X_train, y_train)


def evaluate(model, X_eval, name):
    preds = model.predict(X_eval)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


results = [
    evaluate(log_reg, X_test_lr, "logistic_regression"),
    evaluate(tree, X_test, "decision_tree"),
]
best = max(results, key=lambda r: r["f1"])

# Save metrics
metrics_path = out_dir / "metrics.json"
metrics_path.write_text(json.dumps({r["model"]: r for r in results}, indent=2))

# Confusion matrix plot (best model)
cm = best["confusion_matrix"]
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_title(f"Confusion Matrix ({best['model']})")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
(out_dir / "confusion_matrix.png").parent.mkdir(exist_ok=True, parents=True)
fig.savefig(out_dir / "confusion_matrix.png")
plt.close(fig)

# Feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].barh(feature_cols, log_reg.coef_[0])
axes[0].set_title("Logistic Regression Coefficients")
axes[0].set_xlabel("Coefficient")

axes[1].barh(feature_cols, tree.feature_importances_)
axes[1].set_title("Decision Tree Importances")
axes[1].set_xlabel("Importance")

fig.tight_layout()
fig.savefig(out_dir / "feature_importance.png")
plt.close(fig)

# Summary
total_rows = len(df)
class_counts = y.value_counts().to_dict()
test_counts = y_test.value_counts().to_dict()
print(f"ML dataset: {ml_path}")
print(f"Rows: {total_rows} | Class counts: {class_counts} | Test counts: {test_counts}")
print(f"Best model: {best['model']} | Accuracy: {best['accuracy']:.3f} | F1: {best['f1']:.3f}")
