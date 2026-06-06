import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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

# ── Train XGBoost ─────────────────────────────────────────────
model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    reg_lambda=5,
    scale_pos_weight=1,
    # Boost draw class
    sample_weight=None,
    random_state=42,
    verbosity=1,
    use_label_encoder=False,
    eval_metric='mlogloss',
)

# Class weights to boost draw
sample_weights = y_train.map({0: 1.5, 1: 3.0, 2: 1.5}).values

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# ── Evaluate ──────────────────────────────────────────────────
pred = model.predict(X_test)
acc  = accuracy_score(y_test, pred)

print(f"\n{'='*50}")
print(f"XGBoost Accuracy on 2018+2022: {acc:.4f} ({acc:.1%})")
print(f"{'='*50}")
print(f"\nClassification Report:")
print(classification_report(y_test, pred,
      target_names=["Away Win", "Draw", "Home Win"]))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# ── Feature Importance ────────────────────────────────────────
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
print(importance.head(20).to_string())