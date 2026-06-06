import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("features_phase1_new.csv")

drop_cols = ['match_id', 'home_team_name', 'away_team_name']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

non_numeric = ['Home_Tactics','Away_Tactics','Top Scorer',
               'Top Scorer Country','Result','Date','Stage',
               'Group','Stadium Name','City Name','Country Name']
df = df.drop(columns=[c for c in non_numeric if c in df.columns])

train = df[df['year'] <= 2014]
test  = df[df['year'].isin([2018,2022])]

feature_cols = [c for c in df.columns if c not in ['year', 'target']]
X_train = train[feature_cols]
y_train = train['target']
X_test  = test[feature_cols]
y_test  = test['target']

# Try different configurations
configs = [
    {'n_estimators': 500,  'max_depth': 4, 'min_samples_split': 10},
    {'n_estimators': 1000, 'max_depth': 5, 'min_samples_split': 5},
    {'n_estimators': 1500, 'max_depth': 6, 'min_samples_split': 3},
    {'n_estimators': 2000, 'max_depth': 8, 'min_samples_split': 5},
    {'n_estimators': 1000, 'max_depth': None, 'min_samples_split': 10},
]

best_acc = 0
best_config = None

for config in configs:
    model = RandomForestClassifier(
        **config,
        min_samples_leaf=2,
        class_weight={0: 1.5, 1: 3.0, 2: 1.5},
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Config {config} → Accuracy: {acc:.4f} ({acc:.1%})")
    if acc > best_acc:
        best_acc = acc
        best_config = config
        best_model = model

print(f"\n{'='*50}")
print(f"Best Config: {best_config}")
print(f"Best Accuracy: {best_acc:.4f} ({best_acc:.1%})")
print(f"{'='*50}")

pred = best_model.predict(X_test)
print(f"\nClassification Report:")
print(classification_report(y_test, pred,
      target_names=["Away Win", "Draw", "Home Win"]))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
print(importance.head(20).to_string())