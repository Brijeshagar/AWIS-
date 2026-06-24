"""
AWIS - Phase 1: XGBoost Rejection Prediction Model (Improved)
==============================================================
Improvements over baseline:
  1. Feature scaling analysis  -- XGBoost is tree-based (split-threshold logic),
     so StandardScaler / MinMaxScaler has NO effect on predictions. Scaling is
     intentionally skipped. If a distance-based model (SVM, KNN, LR) is added
     later, apply sklearn Pipeline with StandardScaler at that stage.
  2. Hyperparameter tuning via RandomizedSearchCV (scoring='roc_auc') on a
     stratified 20% sub-sample for speed, then retrain best config on the full
     training set with XGBoost early stopping to find optimal n_estimators.
  3. Before/After comparison printed at the end.

Outputs:
  model.pkl               - Best XGBoost model (pickled)
  feature_importance.json - Feature importance scores (gain)
  confusion_matrix.png    - Confusion-matrix heatmap
  feature_importance.png  - Top-20 feature importance bar chart
  tuning_results.json     - All RandomizedSearchCV trial results
"""

import io
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

# Force UTF-8 stdout so prints work safely on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE      = 42
TEST_SIZE         = 0.20
TUNE_SAMPLE_FRAC  = 0.20   # fraction of train set used for hyperparam search
N_ITER            = 25     # RandomizedSearchCV iterations
CV_FOLDS          = 3      # CV folds during tuning
EARLY_STOP_ROUNDS = 30     # stop adding trees when val-AUC stops improving

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
DATA_PATH     = BASE_DIR / "ml_features.csv"
MODEL_PATH    = BASE_DIR / "model.pkl"
FI_JSON       = BASE_DIR / "feature_importance.json"
TUNING_JSON   = BASE_DIR / "tuning_results.json"
CM_PNG        = BASE_DIR / "confusion_matrix.png"
FI_PNG        = BASE_DIR / "feature_importance.png"

# ===========================================================================
# 1. Load & Analyse
# ===========================================================================
print("=" * 65)
print("  AWIS - Phase 1: XGBoost Rejection Model Training (Improved)")
print("=" * 65)

print("\n[1/7] Loading dataset ...")
df = pd.read_csv(DATA_PATH)

TARGET   = "rejected"
FEATURES = [c for c in df.columns if c != TARGET]

print(f"  Rows     : {len(df):,}")
print(f"  Features : {len(FEATURES)}")
print(f"  Target   : '{TARGET}'")
print(f"\n  Columns  : {FEATURES}")

print("\n  Data types:")
print(df[FEATURES].dtypes.to_string())

missing       = df.isnull().sum()
total_missing = int(missing.sum())
print(f"\n  Missing values: {total_missing} total")
if total_missing > 0:
    print(missing[missing > 0].to_string())
else:
    print("  [OK] No missing values found.")

# Feature scaling note
print("\n  [NOTE] Feature Scaling:")
print("  XGBoost builds decision trees using split thresholds, not")
print("  distances. Scaling (StandardScaler/MinMaxScaler) has zero")
print("  effect on tree splits and is intentionally skipped here.")

class_dist = df[TARGET].value_counts().sort_index()
total      = len(df)
print(f"\n  Class distribution ({TARGET}):")
for label, count in class_dist.items():
    label_str = "Rejected" if label == 1 else "Approved"
    print(f"    {label} ({label_str}): {count:,}  ({count / total * 100:.1f}%)")

neg_count        = class_dist.get(0, 0)
pos_count        = class_dist.get(1, 0)
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
print(f"\n  scale_pos_weight (neg/pos): {scale_pos_weight:.4f}")

# ===========================================================================
# 2. Split 80 / 20
# ===========================================================================
print("\n[2/7] Splitting dataset (80 train / 20 test, stratified) ...")
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"  Train samples : {len(X_train):,}")
print(f"  Test  samples : {len(X_test):,}")

# ===========================================================================
# 3. Baseline Model (for before/after comparison)
# ===========================================================================
print("\n[3/7] Training baseline model ...")
t0 = time.time()

baseline = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
baseline.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

base_pred       = baseline.predict(X_test)
base_proba      = baseline.predict_proba(X_test)[:, 1]
base_accuracy   = accuracy_score(y_test, base_pred)
base_precision  = precision_score(y_test, base_pred, zero_division=0)
base_recall     = recall_score(y_test, base_pred, zero_division=0)
base_f1         = f1_score(y_test, base_pred, zero_division=0)
base_roc_auc    = roc_auc_score(y_test, base_proba)
base_time       = time.time() - t0

print(f"  [OK] Baseline trained in {base_time:.1f}s")
print(f"  Baseline ROC-AUC : {base_roc_auc:.4f}")

# ===========================================================================
# 4. Hyperparameter Tuning (RandomizedSearchCV on sub-sample)
# ===========================================================================
print(f"\n[4/7] Hyperparameter tuning ...")
print(f"  Strategy  : RandomizedSearchCV  (n_iter={N_ITER}, cv={CV_FOLDS})")
print(f"  Scoring   : roc_auc")
print(f"  Sample    : {TUNE_SAMPLE_FRAC*100:.0f}% of train set for speed")

# Stratified sub-sample for tuning speed on large dataset
sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - TUNE_SAMPLE_FRAC), random_state=RANDOM_STATE)
tune_idx, _ = next(sss.split(X_train, y_train))
X_tune = X_train.iloc[tune_idx]
y_tune = y_train.iloc[tune_idx]
print(f"  Tune rows : {len(X_tune):,} (stratified sub-sample)")

param_dist = {
    "n_estimators"    : [200, 300, 400, 500],
    "max_depth"       : [4, 5, 6, 7, 8],
    "learning_rate"   : [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
    "subsample"       : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma"           : [0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha"       : [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda"      : [0.5, 1.0, 1.5, 2.0, 3.0],
}

tuner_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    use_label_encoder=False,
)

search = RandomizedSearchCV(
    estimator=tuner_base,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring="roc_auc",
    cv=CV_FOLDS,
    refit=False,           # we retrain manually with early stopping
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

t1 = time.time()
search.fit(X_tune, y_tune)
tune_time = time.time() - t1

best_params = search.best_params_
print(f"\n  [OK] Tuning complete in {tune_time:.1f}s")
print(f"  Best CV ROC-AUC  : {search.best_score_:.4f}")
print("  Best params:")
for k, v in sorted(best_params.items()):
    print(f"    {k}: {v}")

# Save all trial results
cv_results = pd.DataFrame(search.cv_results_)
tuning_output = {
    "best_params"   : best_params,
    "best_cv_roc_auc": float(search.best_score_),
    "all_trials"    : cv_results[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score").to_dict(orient="records"),
}
with open(TUNING_JSON, "w") as f:
    json.dump(tuning_output, f, indent=2, default=str)
print(f"  [OK] Tuning results saved -> {TUNING_JSON}")

# ===========================================================================
# 5. Retrain Best Model on Full Train Set with Early Stopping
# ===========================================================================
print("\n[5/7] Retraining best config on full train set ...")
print("  Using early stopping (val-AUC) to find optimal n_estimators ...")

# Use a generous upper bound; early stopping will find the real optimum
best_params_final = {k: v for k, v in best_params.items() if k != "n_estimators"}

model = XGBClassifier(
    **best_params_final,
    n_estimators=1000,                   # upper bound; early stopping controls actual
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",                   # optimise for AUC during early stopping
    early_stopping_rounds=EARLY_STOP_ROUNDS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

best_n_estimators = model.best_iteration + 1
print(f"  [OK] Early stopping found optimal n_estimators = {best_n_estimators}")

# ===========================================================================
# 6. Save Model
# ===========================================================================
print("\n[6/7] Saving model ...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"  [OK] Model saved -> {MODEL_PATH}")

# ===========================================================================
# 7. Evaluate & Compare
# ===========================================================================
print("\n[7/7] Evaluating tuned model ...")
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

# Before / After table
print("\n" + "=" * 65)
print("  BEFORE vs AFTER TUNING")
print("=" * 65)
print(f"  {'Metric':<14} {'Baseline':>12} {'Tuned':>12} {'Delta':>10}")
print("-" * 55)

metrics = [
    ("Accuracy",  base_accuracy,  accuracy),
    ("Precision", base_precision, precision),
    ("Recall",    base_recall,    recall),
    ("F1 Score",  base_f1,        f1),
    ("ROC-AUC",   base_roc_auc,   roc_auc),
]
for name, before, after in metrics:
    delta = after - before
    arrow = "+" if delta >= 0 else ""
    print(f"  {name:<14} {before:>12.4f} {after:>12.4f} {arrow}{delta:>9.4f}")

print("=" * 65)

print("\n  Full Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred, target_names=["Approved", "Rejected"]))

# ===========================================================================
# Plots — Confusion Matrix
# ===========================================================================
print("Generating plots ...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Approved", "Rejected"],
    yticklabels=["Approved", "Rejected"],
    ax=ax,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Confusion Matrix - AWIS Rejection Model (Tuned)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
plt.tight_layout()
fig.savefig(CM_PNG, dpi=150)
plt.close(fig)
print(f"  [OK] Confusion matrix saved -> {CM_PNG}")

# ===========================================================================
# Plots — Feature Importance
# ===========================================================================
importance_scores = model.get_booster().get_score(importance_type="gain")
full_importance   = {feat: importance_scores.get(feat, 0.0) for feat in FEATURES}
sorted_importance = dict(sorted(full_importance.items(), key=lambda x: x[1], reverse=True))

with open(FI_JSON, "w") as f:
    json.dump(sorted_importance, f, indent=2)
print(f"  [OK] Feature importance saved -> {FI_JSON}")

top_n     = min(20, len(FEATURES))
fi_series = pd.Series(sorted_importance).head(top_n).sort_values()
palette   = plt.cm.RdYlGn(np.linspace(0.2, 0.9, top_n))

fig2, ax2 = plt.subplots(figsize=(10, 7))
bars = ax2.barh(fi_series.index, fi_series.values, color=palette, edgecolor="white", height=0.7)
ax2.set_title(f"Top-{top_n} Feature Importance (Gain) - AWIS Tuned Model",
              fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Gain Score", fontsize=11)
ax2.set_ylabel("Feature", fontsize=11)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="x", linestyle="--", alpha=0.4)

for bar in bars:
    width = bar.get_width()
    ax2.text(
        width * 1.01, bar.get_y() + bar.get_height() / 2,
        f"{width:,.1f}", va="center", ha="left", fontsize=8.5,
    )

plt.tight_layout()
fig2.savefig(FI_PNG, dpi=150)
plt.close(fig2)
print(f"  [OK] Feature importance plot saved -> {FI_PNG}")

# ===========================================================================
# Final Summary
# ===========================================================================
print("\n" + "=" * 65)
print("  TRAINING COMPLETE - Final Summary")
print("=" * 65)
print(f"  Model              : {MODEL_PATH.name}")
print(f"  Best n_estimators  : {best_n_estimators}  (via early stopping)")
print(f"  Features           : {len(FEATURES)}")
print(f"  Train rows         : {len(X_train):,}")
print(f"  Test  rows         : {len(X_test):,}")
print(f"  Accuracy           : {accuracy * 100:.2f}%")
print(f"  Precision          : {precision:.4f}")
print(f"  Recall             : {recall:.4f}")
print(f"  F1 Score           : {f1:.4f}")
print(f"  ROC-AUC            : {roc_auc:.4f}  (was {base_roc_auc:.4f})")
print(f"  Top feature        : {list(sorted_importance.keys())[0]}")
print("=" * 65)
