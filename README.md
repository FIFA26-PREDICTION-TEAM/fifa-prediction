# FIFA Match Predictor

A Streamlit application for predicting international football matches, with special 2026 World Cup context. The project combines historical international results, rankings, tournament history, squad data, player context, curated 2026 friendlies, and a live 2026 World Cup match feed.

The main app entry point is `app.py`. The model code lives under `model/`, and data loaders live under `data/`.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Or Update Data Files

Core files are loaded from the project root or from `data/processed/` when processed historical data is enabled.

| File | Purpose | Required |
| --- | --- | --- |
| `matches.csv` or `data/processed/matches_all.csv` | Historical international match results | Yes |
| `goalscorers.csv` or `data/processed/goalscorers_all.csv` | Goalscorer history | Optional |
| `shootouts.csv` or `data/processed/shootouts_all.csv` | Penalty shootout outcomes | Optional |
| `rankings.csv` | FIFA ranking history | Optional |
| `player_appearances.csv`, `player_goals.csv`, `award_winners.csv`, `substitutions.csv` | Player-level features | Optional |
| `data/external/fifa_pre_world_cup_friendlies_2026.csv` | Curated 2026 pre-World-Cup friendlies | Optional |
| `data/external/world_cup_2026/FIFA_World_Cup_2026_Teams.csv` | Qualified teams, ranking snapshot, team profile context | Optional |
| `data/external/world_cup_2026/fifa_world_cup_2026_squads.csv` | Official 2026 squads and player context | Optional |
| `data/external/world_cup_2026/wc2026_matches.csv` | Repo fallback copy of ongoing 2026 World Cup matches | Optional |
| `/Users/suhaylahmed/Downloads/wc2026_matches.csv` | Preferred live 2026 World Cup match feed | Optional |

Missing optional files degrade feature coverage and confidence, but the app should continue running.

### 3. Train The Model

```bash
python -m model.train
```

Training saves artifacts to `model/artifacts/`:

- `model.pkl`
- `feature_columns.json`
- `meta.json`
- goal-model artifacts when generated

### 4. Run The App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. If trained artifacts are missing, the app can train inline on first launch.

## Recent Data Updates

### 2026 Pre-World-Cup Friendlies

The curated friendlies file was updated from:

```text
/Users/suhaylahmed/Downloads/until_9_june_fifa_pre_world_cup_friendlies_2026.csv
```

into:

```text
data/external/fifa_pre_world_cup_friendlies_2026.csv
```

That file contains match-level 2026 friendlies through June 9, 2026. It is loaded by `data/international_friendlies.py`, normalized into the project match schema, deduplicated by date/team pairing, and appended to the match history at runtime by `append_friendlies_matches()`.

Friendlies affect the system in two ways:

- They can enter recent-form and match-history features when included in runtime match history.
- They produce aggregate friendlies context such as points per game, goals per match, conceded goals per match, league average goals, and home-advantage style summary values.

Friendlies are excluded from model training by default unless `ML_PRJCT_INCLUDE_FRIENDLIES_TRAIN=1` is set. This keeps training primarily competitive-match based while still allowing friendlies to inform prediction context.

### Live 2026 World Cup Match Feed

The ongoing World Cup feed is expected at:

```text
/Users/suhaylahmed/Downloads/wc2026_matches.csv
```

That Downloads file is preferred automatically. A fallback copy is also stored at:

```text
data/external/world_cup_2026/wc2026_matches.csv
```

The path can be overridden with:

```bash
export ML_PRJCT_WORLD_CUP_2026_MATCHES_PATH=/path/to/wc2026_matches.csv
```

Because this CSV will be updated frequently during the tournament, the live-data loaders use a short Streamlit cache TTL. The default is 60 seconds:

```bash
export ML_PRJCT_LIVE_DATA_CACHE_TTL_SECONDS=60
```

The live feed is parsed by `data/world_cup_2026.py` and appended into the main match table by `data/ingest.py`. That means completed 2026 World Cup matches now affect:

- Recent form
- Head-to-head
- World Cup match history
- Goals scored/conceded trends
- Clean sheet rates
- Current tournament team performance
- The final 2026 World Cup context probability adjustment

## How Prediction Works

### 1. Data Loading

`data/ingest.py` builds the data bundle used by the app:

- Historical matches
- Goalscorers
- Shootouts
- FIFA rankings
- Substitutions and player files
- Copa America 2024 context
- EURO 2024 context
- 2026 friendlies context
- 2026 World Cup team, squad, and live match context

`load_matches()` now appends both curated 2026 friendlies and completed 2026 World Cup matches before feature generation.

### 2. Leakage-Safe Feature Generation

All core features are built in `data/preprocess.py` through `build_feature_row()`.

Training uses a rolling `as_of_date`: for each historical match, features only use data before that match. This prevents future results from leaking into the training row.

Prediction uses the latest available match date plus one day. This is intentional for the live World Cup CSV because the feed has dates but not kickoff times. Same-day completed matches can therefore be included in predictions after the CSV is updated.

### 3. Main Feature Groups

The model uses these feature families:

- Head-to-head history: meetings, wins, draws, World Cup meetings, last-five trends, weighted win ratio
- Recent competitive form: last 10 competitive matches, goals scored, goals conceded, clean sheets, win streaks, unbeaten streaks
- Tournament history: World Cup titles, finals reached, knockout strength, penalty shootout records
- FIFA rankings: rank, ranking points, rank difference
- Goalscoring context: World Cup goals-per-game and scoring-first win rate
- Super-sub context: late-goal substitute heuristic
- Copa America 2024 context: match and aggregate tournament features
- EURO 2024 context: player-strength and team-performance snapshot features
- 2026 friendlies context: pre-World-Cup form and aggregate friendly performance
- Raw CatBoost context: team A, team B, normalized tournament category

### 4. Model Architecture

The project uses CatBoost because the problem is tabular and contains many categorical signals: teams, tournaments, venues, squad context, and competition type.

The saved production artifact is a two-stage CatBoost result model when enabled:

- Stage 1 estimates draw probability.
- Stage 2 estimates non-draw direction.
- A calibrated draw threshold is selected from previous World Cup-style validation when enough data exists.

The final output classes are:

- `0`: team B wins
- `1`: draw
- `2`: team A wins

At prediction time, the engine builds the match in both orientations:

1. Team A vs Team B
2. Team B vs Team A

It averages the mirrored probabilities so the result is less sensitive to home/away or input ordering when the venue is neutral.

### 5. 2026 World Cup Squad Context

`data/world_cup_2026.py` loads team and squad data, normalizes team names, and builds 2026 profiles.

Each team profile can include:

- Group
- Confederation
- FIFA ranking snapshot
- Best World Cup finish
- World Cup debut flag
- Curated key players
- Official squad summary
- Squad age profile
- Squad height profile
- Top-league share
- Club-strength score
- Squad-depth score
- Current tournament performance from `wc2026_matches.csv`

The squad-strength score blends ranking, history, key-player strength, confederation strength, and debut penalty. Official squad depth is then blended into the profile strength.

### 6. Live Tournament Performance Adjustment

After the base model probability is produced, `adjust_probabilities_for_2026_context()` applies a small, bounded 2026 World Cup context adjustment.

The adjustment has two parts:

- Squad edge: based on official squad/profile strength differences.
- Current tournament edge: based on live 2026 World Cup results and match stats.

The live tournament score uses:

- Points per match
- Goal difference per match
- Goals for per match
- Goals against per match
- Shots-on-target difference or shot difference
- Possession average

The current-tournament adjustment is sample-weighted. Early in the tournament, one match has limited influence. As teams play up to three matches, the live adjustment reaches full sample weight. The final probability shift is clipped so a small live sample cannot overwhelm the trained model.

The context object returned by prediction includes:

- `squad_probability_shift`
- `current_tournament_applied`
- `team_a_current_tournament`
- `team_b_current_tournament`
- `current_tournament_score_delta`
- `current_tournament_sample_weight`
- `current_tournament_probability_shift`
- Final team A, draw, and team B probability shifts

### 7. Confidence Score

The app reports confidence using:

```text
confidence = (max_probability * 0.7) + (data_coverage * 0.2) + (h2h_bonus up to 0.1)
```

Confidence is clipped to `[0, 1]` and labeled:

- Very High
- High
- Medium
- Low

## Training Defaults And Environment Switches

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ML_PRJCT_USE_HISTORICAL_WEIGHTED` | `1` | Use broader historical processed data and era/tournament weighting |
| `ML_PRJCT_MIN_DATA_YEAR` | `1872` with historical mode, otherwise `2006` | Earliest match year to load |
| `ML_PRJCT_MAX_DATA_DATE` | Today | Latest date allowed into loaded data |
| `ML_PRJCT_USE_PROCESSED` | `0` | Prefer `data/processed/matches_<year>.csv` style files |
| `ML_PRJCT_INCLUDE_COPA` | `1` | Include Copa America context |
| `ML_PRJCT_INCLUDE_FRIENDLIES` | `1` | Include friendlies context |
| `ML_PRJCT_INCLUDE_FRIENDLIES_TRAIN` | `0` | Include curated friendlies as training targets |
| `ML_PRJCT_WORLD_CUP_2026_MATCHES_PATH` | Downloads CSV if present | Override live World Cup match feed path |
| `ML_PRJCT_LIVE_DATA_CACHE_TTL_SECONDS` | `60` | Cache TTL for live World Cup data and match loading |
| `ML_PRJCT_DRAW_CLASS_WEIGHT` | `1.6` | Multiclass draw weight |
| `ML_PRJCT_DRAW_PROB_THRESHOLD` | `0.24` | Default draw decision threshold |
| `ML_PRJCT_USE_TWO_STAGE_RESULT` | `1` | Enable two-stage CatBoost result model |
| `ML_PRJCT_TRAIN_SAMPLE_SIZE` | unset | Optional quick experiment sample size |

## Super-Sub Logic

The historical datasets do not always contain reliable lineup data. The system therefore infers probable substitute impact heuristically.

A player is counted as a probable substitute when they score after the 60th minute in a match and did not score earlier in that same match. A team gets the binary super-sub flag when it has enough repeated late substitute-style scoring impact.

This is intentionally approximate. Real lineup and substitution data would be more accurate when available.

## Goal Prediction

`model/goals.py` trains CatBoost regressors for expected goals-style score prediction. It uses historical scoring patterns, ranking context, venue mode, and team form. The final displayed outcome uses a calibrated blend of the result classifier probabilities and the goal model's aggregate win/draw/loss probabilities. The displayed scoreline is then selected from the goal model's best scoreline for that same final outcome, so a win label cannot be paired with a draw or loss scoreline.

## Project Structure

```text
fifa/
+-- app.py
+-- README.md
+-- requirements.txt
+-- data/
|   +-- ingest.py
|   +-- preprocess.py
|   +-- international_friendlies.py
|   +-- world_cup_2026.py
|   +-- external/
|   |   +-- fifa_pre_world_cup_friendlies_2026.csv
|   |   +-- world_cup_2026/
|   |       +-- FIFA_World_Cup_2026_Teams.csv
|   |       +-- fifa_world_cup_2026_squads.csv
|   |       +-- non_world_cup_2026_squads.csv
|   |       +-- wc2026_matches.csv
|   +-- processed/
+-- model/
|   +-- features.py
|   +-- train.py
|   +-- predict.py
|   +-- goals.py
|   +-- artifacts/
+-- tests/
+-- utils/
```

## Validation

Run tests with the project root on `PYTHONPATH`:

```bash
PYTHONPATH=. pytest -q
```

Plain `pytest -q` may fail in this checkout if local packages such as `data` and `model` are not on the import path.

## Deployment Notes

For Streamlit Cloud or another hosted deployment:

1. Push the repository.
2. Install dependencies from `requirements.txt`.
3. Set `app.py` as the Streamlit entry point.
4. Include required CSVs or mount them from external storage.
5. For a live World Cup feed, set `ML_PRJCT_WORLD_CUP_2026_MATCHES_PATH` to the mounted CSV path.

The app is designed to degrade gracefully when optional context files are unavailable, but predictions are strongest when rankings, squads, friendlies, and the live World Cup feed are present.
