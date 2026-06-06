import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Load features ──────────────────────────────────────────────
df = pd.read_csv("features_phase1_new.csv")

# ── Drop identifier columns ───────────────────────────────────
drop_cols = ['match_id', 'home_team_name', 'away_team_name']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ── Drop non-numeric columns ──────────────────────────────────
non_numeric = ['Home_Tactics','Away_Tactics','Top Scorer',
               'Top Scorer Country','Result','Date','Stage',
               'Group','Stadium Name','City Name','Country Name']
df = df.drop(columns=[c for c in non_numeric if c in df.columns])

# ── Time based split ──────────────────────────────────────────
train = df[df['year'] <= 2014]
test  = df[df['year'].isin([2018, 2022])]

print(f"Train rows: {len(train)}")
print(f"Test rows:  {len(test)}")
print(f"Features:   {len(df.columns) - 2}")

# ── Features and target ───────────────────────────────────────
feature_cols = [c for c in df.columns if c not in ['year', 'target']]
X_train = train[feature_cols]
y_train = train['target']
X_test  = test[feature_cols]
y_test  = test['target']

print(f"\nTarget distribution in train:\n{y_train.value_counts()}")
print(f"\nTarget distribution in test:\n{y_test.value_counts()}")

# ── Train CatBoost ────────────────────────────────────────────
model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    class_weights=[1.5, 3.0, 1.5],
    random_seed=42,
    verbose=100,
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
)

# ── Evaluate ──────────────────────────────────────────────────
pred = model.predict(X_test)
acc  = accuracy_score(y_test, pred)

print(f"\n{'='*50}")
print(f"CatBoost Accuracy on 2018+2022: {acc:.4f} ({acc:.1%})")
print(f"{'='*50}")
print(f"\nClassification Report:")
print(classification_report(y_test, pred,
      target_names=["Away Win", "Draw", "Home Win"]))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# ── Feature Importance ────────────────────────────────────────
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.get_feature_importance()
}).sort_values('importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
print(importance.head(20).to_string())