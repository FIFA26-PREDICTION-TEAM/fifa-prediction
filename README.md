# FIFA Match Predictor

A machine-learning-powered international football match prediction system built entirely in Streamlit.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your data files

Drop the following CSV files into the **project root** (same folder as `app.py`):

| File | Source | Required? |
|------|--------|-----------|
| `matches.csv` | Kaggle "International football results from 1872 to 2024" | **Required** |
| `goalscorers.csv` | Same Kaggle dataset | Optional |
| `shootouts.csv` | Same Kaggle dataset | Optional |
| `rankings.csv` | Kaggle "FIFA World Rankings 1992–2024" | Optional |
| `data/external/international-copa-america-*-2024-to-2024-stats.csv` | Copa America 2024 stats exports | Optional |

Missing files produce a `st.warning()` in the app and reduce the confidence score proportionally — the app does not crash.

### 3. Train the model (run once)

```bash
python -m model.train
```

This builds leakage-safe rolling features, trains CatBoost, and saves the validated model to `model/artifacts/`. CatBoost is the fixed model for this project because the data is tabular football data with many categorical signals such as teams, tournaments, venues, squads, and match context.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`. If no trained model is found, it trains one inline on first launch.

---

## How it works

### Training approach

The model is trained on competitive international matches from 2006 onwards by default. Features are built with a **rolling as-of-date window** — each training row only uses data available before that match, preventing data leakage.

Validation is time-based: the newest match year is held out so future-style predictions are measured against past-only training data. Symmetric team-swap augmentation is still used, but both orientations of the same match stay in the same time fold.

The project uses **CatBoost only**. Validation still reports accuracy, balanced accuracy, and probability log loss, but the modeling strategy is fixed so future work focuses on adding better data and safer features rather than switching algorithms.

When the Copa America 2024 files are present, the loader appends the 32 Copa match results at runtime without modifying `matches.csv`. Pre-match odds/PPG/xG are eligible match-context features. Full tournament team/player/league aggregates are treated as post-tournament context so they are not used to predict matches that occurred before the tournament finished.

### Features

- **Head-to-head**: historical win/draw/loss record, World Cup meetings, last-5 trends, weighted win ratio
- **Recent form**: wins/draws/losses, goals scored/conceded, clean sheet rate, streaks (last 10 competitive matches)
- **Tournament history**: World Cup titles, finals reached, knockout win rate, penalty shootout record
- **FIFA rankings**: rank, ranking points, rank differential
- **Goalscoring**: WC goals-per-game average, scoring-first win rate
- **Super-sub**: binary flag per team (see below)
- **Raw context for CatBoost**: team A, team B, normalized tournament category

### Super-sub detection

Since match lineup data is absent from the Kaggle dataset, substitute appearances are **inferred heuristically**: a player who scores only after the 60th minute in a match (with no earlier-minute goals in that same match) is counted as a probable substitute appearance.

Threshold: 3+ inferred substitute appearances with a goals-as-sub rate > 40%.

⚠️ These numbers are estimates — exact lineup data would be required for precision.

### Confidence score

```
confidence = (max_prob × 0.7) + (data_coverage × 0.2) + (h2h_bonus up to 0.1)
```

Clipped to [0, 1]. Labels: Very High (>80%), High (>65%), Medium (>50%), Low.

---

## Streamlit Cloud deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set main file to `app.py`
4. Add your CSV files to the repo (or use Streamlit secrets + cloud storage for large files)
5. Deploy — Streamlit Cloud runs `pip install -r requirements.txt` automatically

> Note: `catboost` is required by `requirements.txt` because it is the single production model.

---

## Project structure

```
fifa_predictor/
├── app.py                   # Streamlit UI — sole entry point
├── requirements.txt
├── data/
│   ├── ingest.py            # CSV loaders with @st.cache_data
│   └── preprocess.py        # Feature engineering (build_feature_row)
├── model/
│   ├── features.py          # Feature column list, tournament weights, recency decay
│   ├── train.py             # Training pipeline
│   ├── predict.py           # Inference engine
│   └── artifacts/           # model.pkl, scaler.pkl, feature_columns.json, meta.json
└── utils/
    └── supersub.py          # Super-sub detection
```
