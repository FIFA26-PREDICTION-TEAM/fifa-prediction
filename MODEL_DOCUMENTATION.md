# FIFA World Cup 2026 — Match Prediction Model: Full Technical Documentation

> **Audience:** Developers, data scientists, or anyone who wants to understand exactly how this prediction system works — the data, the features, the algorithms, the inference pipeline, and the design decisions behind them.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & File Structure](#2-architecture--file-structure)
3. [Data Sources & Ingestion](#3-data-sources--ingestion)
4. [Feature Engineering](#4-feature-engineering)
   - 4.1 [Venue Context](#41-venue-context)
   - 4.2 [Head-to-Head (H2H) Features](#42-head-to-head-h2h-features)
   - 4.3 [Recent Team Form](#43-recent-team-form)
   - 4.4 [Tournament History](#44-tournament-history)
   - 4.5 [FIFA Rankings](#45-fifa-rankings)
   - 4.6 [Draw-Tendency & Team Similarity Signals](#46-draw-tendency--team-similarity-signals)
   - 4.7 [Goalscoring Features](#47-goalscoring-features)
   - 4.8 [Super-Sub Detection](#48-super-sub-detection)
   - 4.9 [Copa America 2024 Context](#49-copa-america-2024-context)
   - 4.10 [International Friendlies 2026 Context](#410-international-friendlies-2026-context)
   - 4.11 [EURO 2024 Player-Strength Snapshot](#411-euro-2024-player-strength-snapshot)
   - 4.12 [Player Impact Features](#412-player-impact-features)
   - 4.13 [Raw Categorical Context (CatBoost-native)](#413-raw-categorical-context-catboost-native)
5. [The No-Leakage Guarantee](#5-the-no-leakage-guarantee)
6. [Training Pipeline](#6-training-pipeline)
   - 6.1 [Data Filtering & Match Selection](#61-data-filtering--match-selection)
   - 6.2 [Sample Weighting](#62-sample-weighting)
   - 6.3 [Data Augmentation — Symmetric Team Swap](#63-data-augmentation--symmetric-team-swap)
   - 6.4 [Time-Based Validation Split](#64-time-based-validation-split)
   - 6.5 [Historical Source Capping](#65-historical-source-capping)
7. [The Machine Learning Models](#7-the-machine-learning-models)
   - 7.1 [Why CatBoost?](#71-why-catboost)
   - 7.2 [Result Classifier — Two-Stage Architecture](#72-result-classifier--two-stage-architecture)
   - 7.3 [Draw Calibration](#73-draw-calibration)
   - 7.4 [Goal Prediction Model](#74-goal-prediction-model)
8. [Inference Pipeline (Prediction)](#8-inference-pipeline-prediction)
   - 8.1 [Symmetry Averaging](#81-symmetry-averaging)
   - 8.2 [2026 World Cup Context Adjustment](#82-2026-world-cup-context-adjustment)
   - 8.3 [Label Decision Policy](#83-label-decision-policy)
   - 8.4 [Confidence Score](#84-confidence-score)
9. [All Parameters & Configuration](#9-all-parameters--configuration)
10. [Full Feature Column Reference](#10-full-feature-column-reference)
11. [Tournament Weights](#11-tournament-weights)
12. [Model Evaluation Metrics](#12-model-evaluation-metrics)
13. [Known Limitations & Design Decisions](#13-known-limitations--design-decisions)

---

## 1. Project Overview

This is a machine learning system that predicts the outcome of international football matches — specifically tuned for the **FIFA World Cup 2026**. Given two national teams and a venue context, the model outputs:

- **Win / Draw / Loss probabilities** for Team A
- **Expected goals** for each team (a separate regression model)
- **Most likely scoreline** (derived via Poisson distribution)
- **Confidence score** (data coverage × prediction strength)
- **Key factors** (human-readable bullet points driving the prediction)

The system is built on top of:
- **CatBoost** — the sole algorithm for both classification (result) and regression (goals)
- **Streamlit** — for the interactive UI (`app.py`)
- Historical international match data from 1872 to present

---

## 2. Architecture & File Structure

```
ML_PRJCT/
├── app.py                          # Streamlit UI — sole user-facing entry point
├── requirements.txt
│
├── data/
│   ├── ingest.py                   # CSV loaders (cached with @st.cache_data)
│   ├── preprocess.py               # build_feature_row() — all feature engineering
│   ├── copa_america.py             # Copa America 2024 data loader & feature builder
│   ├── euro_2024.py                # EURO 2024 player-stats loader & feature builder
│   ├── international_friendlies.py # 2026 pre-WC friendlies loader & feature builder
│   ├── world_cup_2026.py           # 2026 WC team profiles, squads, probability adjustments
│   ├── processed/                  # Pre-joined match CSVs (optional performance cache)
│   ├── raw/                        # Raw data directory (international_results/)
│   └── external/                   # External enrichment data (Copa, EURO, 2026 squads)
│
├── model/
│   ├── features.py                 # FEATURE_COLUMNS list, tournament weights, recency decay
│   ├── train.py                    # Full training pipeline — train_and_save()
│   ├── predict.py                  # Inference engine — predict_match()
│   ├── goals.py                    # Goal regression model — train_and_save_goals(), predict_goals()
│   └── artifacts/
│       ├── model.pkl               # Serialised model bundle (CatBoost + metadata)
│       ├── scaler.pkl              # Placeholder (backwards compatibility — unused)
│       ├── feature_columns.json    # Ordered feature column names
│       └── meta.json               # Training metadata (accuracy, rows, date range, etc.)
│
├── utils/
│   ├── supersub.py                 # Super-sub detection engine
│   ├── player_stats.py             # Player impact feature computation
│   └── player_achievements.py      # Award extraction helpers
│
└── scripts/
    ├── update_open_data.py          # Helper to refresh raw CSV data
    └── evaluate_world_cup_holdouts.py
```

### Key design principle: Single feature function

`data/preprocess.py::build_feature_row()` is the **single source of truth** for feature computation. It is called identically by:
- `model/train.py` (during training, once per historical match)
- `model/predict.py` (during inference, for the matchup to predict)

This guarantees **no train/serve skew** — the exact same logic computes features in both contexts.

---

## 3. Data Sources & Ingestion

| File | Description | Required? |
|------|-------------|-----------|
| `matches.csv` | All international match results from 1872–present (home team, away team, scores, tournament, date, venue) | **Required** |
| `goalscorers.csv` | Individual goal records (scorer, minute, own-goal flag, penalty flag) | Optional |
| `shootouts.csv` | Penalty shootout winners (for tie-breaking and shootout record features) | Optional |
| `rankings.csv` | FIFA World Ranking snapshots (rank, total points, rank date, country) | Optional |
| `player_appearances.csv` | Player-level appearance records (tournament, position, match ID) | Optional |
| `player_goals.csv` | Player-level goal records (scorer ID, team, match, own-goal) | Optional |
| `award_winners.csv` | Golden Ball / Boot / Glove winners by WC tournament | Optional |
| `data/external/world_cup_2026/FIFA_World_Cup_2026_Teams.csv` | 2026 WC team profiles, key players, rankings | Optional |
| `data/external/world_cup_2026/fifa_world_cup_2026_squads.csv` | Full 2026 squad lists per team | Optional |
| `data/external/international-copa-america-*` | Copa America 2024 stats (odds, xG, PPG, player ratings) | Optional |

Missing files degrade the confidence score proportionally. The app does **not** crash — all optional features default to zero.

All loaders are wrapped in `@st.cache_data` decorators to avoid re-reading CSVs on every Streamlit rerender.

---

## 4. Feature Engineering

All features are computed by `data/preprocess.py::build_feature_row(team_a, team_b, as_of_date, ...)`.

There are **three types** of inputs to the model:

| Type | Description | Examples |
|------|-------------|---------|
| **Categorical** (`RAW_CONTEXT_COLUMNS`) | String labels handled natively by CatBoost's internal embeddings | `team_a`, `team_b`, `tournament` |
| **Numeric** (`FEATURE_COLUMNS`) | 150+ computed numeric features | form stats, rankings, H2H ratios |

Total model input = 3 categorical + 150+ numeric = **153 columns**.

---

### 4.1 Venue Context

Computed from the match metadata (home/away/neutral flags, host country).

| Feature | Description |
|---------|-------------|
| `team_a_is_home` | 1.0 if Team A is the official home team (not a neutral venue) |
| `team_b_is_home` | 1.0 if Team B is the official home team |
| `neutral_venue` | 1.0 if match is at a neutral venue |
| `team_a_host_country` | 1.0 if the match country matches Team A's nation |
| `team_b_host_country` | 1.0 if the match country matches Team B's nation |

> **Why this matters:** Home advantage is one of the strongest predictors in football. The distinction between "official home team" and "host country" captures subtle effects — e.g., a team playing in their own country at a tournament even if listed as "neutral".

For the 2026 World Cup, all predictions default to `neutral_venue = 1.0` since the WC is hosted across US/Canada/Mexico.

---

### 4.2 Head-to-Head (H2H) Features

Source: all historical matches where both teams appeared, filtered to `date < as_of_date`.

The H2H computation applies **two weighting schemes simultaneously**:
1. **Tournament weight** — more recent and prestigious competitions count more (see Section 11)
2. **Recency weight** — linear decay: 1.0 for matches ≤ 2 years old, dropping to 0.4 for matches ≥ 10 years old

#### Recency Decay Formula
```
if years_old ≤ 2:
    weight = 1.0
elif years_old ≥ 10:
    weight = 0.4
else:
    weight = 1.0 − ((years_old − 2) / 8) × 0.6
```

| Feature | Description |
|---------|-------------|
| `h2h_total_meetings` | Total historical meetings between the two teams |
| `h2h_team_a_wins` | Number of Team A wins |
| `h2h_team_b_wins` | Number of Team B wins |
| `h2h_draws` | Number of draws |
| `h2h_wc_only_team_a_wins` | Team A wins in World Cup matches specifically |
| `h2h_wc_only_team_b_wins` | Team B wins in World Cup matches specifically |
| `h2h_last5_team_a_wins` | Team A wins in the last 5 meetings |
| `h2h_last5_goals_a` | Goals scored by Team A in the last 5 meetings |
| `h2h_last5_goals_b` | Goals scored by Team B in the last 5 meetings |
| `h2h_weighted_win_ratio_a` | Tournament-and-recency-weighted win ratio for Team A (0.5 = balanced) |

---

### 4.3 Recent Team Form

Computed from **the last 10 competitive matches** for each team (friendlies excluded).

"Competitive" tournaments include: World Cup, Copa América, UEFA Euro, AFCON, Asian Cup, Gold Cup, Nations League, Qualifiers, Olympics, Confederations Cup.

| Feature | Description |
|---------|-------------|
| `team_a_form_wins` | Wins in last 10 competitive matches |
| `team_a_form_draws` | Draws in last 10 competitive matches |
| `team_a_form_losses` | Losses in last 10 competitive matches |
| `team_a_goals_scored_avg` | Average goals scored per match |
| `team_a_goals_conceded_avg` | Average goals conceded per match |
| `team_a_clean_sheet_rate` | Fraction of matches where Team A kept a clean sheet |
| `team_a_win_streak` | Current consecutive win streak (most-recent-first) |
| `team_a_unbeaten_streak` | Current consecutive unbeaten run (wins + draws) |

All 8 features are mirrored for Team B with `team_b_` prefix.

#### How streaks are computed
Results are scanned backwards from the most recent match. The streak counter increments until the first loss (for win streak) or first loss (for unbeaten streak).

---

### 4.4 Tournament History

World Cup-specific pedigree for each team.

| Feature | Description |
|---------|-------------|
| `team_a_wc_titles` | World Cup titles won (from a hardcoded known-winners dict) |
| `team_a_wc_finals_reached` | WC finals reached (lower bound = title count; exact stage not in dataset) |
| `team_a_wc_win_rate_knockouts` | Overall WC win rate used as a proxy for knockout performance |
| `team_a_penalty_shootout_wins` | Penalty shootout wins (from shootouts.csv) |
| `team_a_penalty_shootout_losses` | Penalty shootout losses |

All 5 features are mirrored for Team B.

> **Design note:** The main `matches.csv` dataset does not include a match-stage column (e.g., "Quarter-Final"). WC titles are stored in a hardcoded dictionary: Brazil=5, Germany=4, Italy=4, Argentina=3, France=2, Uruguay=2, England=1, Spain=1. Knockout win rate is computed directly from WC match results as an approximation.

---

### 4.5 FIFA Rankings

The model uses FIFA/Coca-Cola World Ranking snapshot data for each team.

For each team, the **most recent ranking snapshot on or before `as_of_date`** is retrieved using binary search (`np.searchsorted`) over sorted rank dates.

| Feature | Description |
|---------|-------------|
| `team_a_rank` | FIFA rank (lower = better; 0 = no ranking data found) |
| `team_b_rank` | FIFA rank for Team B |
| `rank_diff` | `team_b_rank − team_a_rank` (positive = Team A is ranked higher) |
| `rank_abs_diff` | Absolute rank difference |
| `rank_closeness` | `1 / (1 + rank_abs_diff / 25.0)` — high when teams are closely ranked |
| `team_a_rank_points` | FIFA ranking points (total points in the ranking system) |
| `team_b_rank_points` | FIFA ranking points for Team B |

> **Rank closeness** is a continuous signal useful for draw prediction — closely ranked teams have higher draw probability.

---

### 4.6 Draw-Tendency & Team Similarity Signals

These features are **derived from the features already computed above**. They explicitly encode signals that are predictive of draws.

| Feature | Description | Formula |
|---------|-------------|---------|
| `h2h_draw_rate` | Historical draw rate between the two teams | `h2h_draws / h2h_total_meetings` |
| `h2h_balance_score` | How evenly matched the teams are historically | `1 − min(|h2h_weighted_win_ratio_a − 0.5| × 2, 1)` |
| `combined_form_draw_rate` | Average draw rate from recent form of both teams | `(team_a_draw_rate + team_b_draw_rate) / 2` |
| `form_draw_rate_diff` | Absolute difference in individual draw rates | — |
| `attack_balance_abs_diff` | Difference in average goals scored | `|team_a_goals_scored_avg − team_b_goals_scored_avg|` |
| `defense_balance_abs_diff` | Difference in average goals conceded | `|team_a_goals_conceded_avg − team_b_goals_conceded_avg|` |
| `combined_clean_sheet_rate` | Average clean sheet rate (proxy for defensive strength) | — |
| `low_scoring_tendency` | Whether both teams tend toward low-scoring games | `clip((2.5 − avg_total_goal_tendency) / 2.5, 0, 1)` |
| `draw_similarity_score` | Composite signal from 6 sub-signals | See below |

#### `draw_similarity_score` Computation

This is a **composite signal** averaging 6 sub-components:
```python
closeness_parts = [
    rank_closeness,                           # Are they similarly ranked?
    1 - min(attack_balance_abs_diff / 3, 1),  # Similar attack strength?
    1 - min(defense_balance_abs_diff / 3, 1), # Similar defense strength?
    h2h_balance_score,                         # Historically even?
    combined_form_draw_rate,                   # Both draw a lot?
    low_scoring_tendency,                      # Low-scoring games expected?
]
draw_similarity_score = mean(closeness_parts)
```

A high `draw_similarity_score` (close to 1.0) signals that the match is structurally similar to past draws.

---

### 4.7 Goalscoring Features

Source: `goalscorers.csv` and WC match results.

| Feature | Description |
|---------|-------------|
| `team_a_avg_goals_wc` | Team A's average goals per match across all World Cup appearances |
| `team_b_avg_goals_wc` | Same for Team B |
| `team_a_scoring_first_win_rate` | Win rate when Team A scores first (Bayesian-smoothed) |
| `team_b_scoring_first_win_rate` | Same for Team B |

#### Bayesian Smoothing (Scoring-First Win Rate)
```python
sf_win_rate = (scoring_first_wins + 2) / (scoring_first_total + 4)
```
This is a Laplace-smoothed ratio with a prior of 0.5 (50% win rate). It prevents extreme values from small samples (e.g., 1/1 = 100%) and naturally regresses to 50% when there's little data.

---

### 4.8 Super-Sub Detection

A "super-sub" is a player who scores an unusually high proportion of their goals as a substitute.

**Primary path (when `substitutions.csv` is available):**
Uses real substitution records. Counts exact substitute appearances and goals scored in those specific matches.

**Fallback path (no substitution data):**
Heuristic proxy — a player who scores only after minute 60 in a match (with no earlier goals in that same match) is treated as a probable substitute appearance.

**Threshold:** ≥ 3 substitute appearances AND goals-as-sub rate > 40%.

| Feature | Type | Description |
|---------|------|-------------|
| `team_a_has_supersub` | Binary (0/1) | 1 if Team A has a qualifying super-sub |
| `team_b_has_supersub` | Binary (0/1) | 1 if Team B has a qualifying super-sub |

The full super-sub record (name, goals, appearances) is also surfaced in the UI as a "key factor" — but the model itself only uses the binary flag.

---

### 4.9 Copa America 2024 Context

If Copa America 2024 data files are present, additional features are loaded for teams that participated.

These are split into two groups:

**Pre-match context** (available before the tournament game being predicted):
- Points-per-game, expected goals (xG) going into that specific match
- Betting odds (win/draw/loss) and implied win probabilities

**Aggregate tournament context** (available after the tournament ended):
- Team-level: PPG, xG for/against, goals per match, conceded per match
- Player-level: average rating, goal rate, xG per 90 minutes
- League-level: average goals, average xG, home advantage factor

| Features (per team prefix `team_a_copa_` / `team_b_copa_`) |
|-------------------------------------------------------------|
| `_pre_match_ppg`, `_pre_match_xg` |
| `_odds_win`, `draw_copa_odds`, `_odds_win` |
| `_implied_win_prob`, `draw_copa_implied_prob` |
| `_team_ppg`, `_team_xg_for`, `_team_xg_against` |
| `_team_goals_per_match`, `_team_conceded_per_match` |
| `_player_avg_rating`, `_player_goal_rate`, `_player_xg_per90` |

League-wide: `copa_league_avg_goals`, `copa_league_xg_avg`, `copa_league_home_advantage`

---

### 4.10 International Friendlies 2026 Context

For pre-World Cup 2026 friendlies, the same structure as Copa America is replicated:

**Pre-match context:** PPG, xG, odds, implied probabilities before each friendly match.
**Aggregate context:** Team and player performance across the entire 2026 friendlies dataset.

| Features (prefix `team_a_friendlies_` / `team_b_friendlies_`) |
|----------------------------------------------------------------|
| `_pre_match_ppg`, `_pre_match_xg` |
| `_odds_win`, `draw_friendlies_odds` |
| `_implied_win_prob`, `draw_friendlies_implied_prob` |
| `_team_ppg`, `_team_xg_for`, `_team_xg_against` |
| `_team_goals_per_match`, `_team_conceded_per_match` |
| `_player_avg_rating`, `_player_goal_rate`, `_player_xg_per90` |

League-wide: `friendlies_league_avg_goals`, `friendlies_league_xg_avg`, `friendlies_league_home_advantage`

---

### 4.11 EURO 2024 Player-Strength Snapshot

A post-EURO 2024 player-performance snapshot is integrated for European teams.

| Feature Group | Features |
|--------------|---------|
| **Squad impact** | `team_a_euro_squad_impact`, `team_a_euro_top5_impact` |
| **Positional impact** | `_euro_attack_impact`, `_euro_midfield_impact`, `_euro_defense_impact`, `_euro_goalkeeping_impact` |
| **Coverage** | `_euro_minutes_coverage` — fraction of squad minutes represented |
| **Team stats** | `_euro_team_ppg`, `_euro_team_goal_diff_per_match`, `_euro_team_goals_per_match`, `_euro_team_conceded_per_match` |
| **Playing style** | `_euro_team_shots_per_match`, `_euro_team_shots_on_target_rate`, `_euro_team_pass_accuracy` |
| **Tactical** | `_euro_team_defensive_actions_per_match`, `_euro_team_set_piece_pressure`, `_euro_team_discipline_risk` |
| **Leader stats** | `_euro_leader_goals_per_match`, `_euro_leader_assists_per_match`, `_euro_leader_goal_contrib_per_match` |

All 22 features above are mirrored for `team_b_euro_*`.

---

### 4.12 Player Impact Features

Computed from player appearance history, goal history, and award records.

| Feature | Description |
|---------|-------------|
| `team_a_top_scorer_wc_goals` | World Cup career goal tally of the team's all-time top scorer |
| `team_a_attack_rating` | Average goals/appearance for attacking-position players (≥2 appearances) |
| `team_a_golden_ball_count` | Number of Golden Ball awards won by team's players |
| `team_a_golden_boot_count` | Number of Golden Boot (top scorer) awards |
| `team_a_golden_glove_count` | Number of Golden Glove (best keeper) awards |
| `team_a_avg_player_experience` | Average number of distinct World Cup tournaments per squad member |

All 6 features are mirrored for Team B.

**Attack positions** used for `attack_rating`: FW, CF, LW, RW, LF, RF, SS, AM.

Data is filtered to post-2006 to avoid retired-player bias (`DATA_FROM_YEAR = 2006`).

---

### 4.13 Raw Categorical Context (CatBoost-native)

Three string features are passed directly to CatBoost as categorical features. CatBoost learns target-encoded embeddings for these automatically during training — no one-hot encoding needed.

| Feature | Values |
|---------|--------|
| `team_a` | Team name string (e.g., "Brazil", "Germany") |
| `team_b` | Team name string |
| `tournament` | Normalised tournament category (e.g., "fifa world cup", "copa america") |

This allows CatBoost to learn **team-specific strengths and biases** beyond what the numeric features capture — for example, that Brazil tends to perform well regardless of their current form.

---

## 5. The No-Leakage Guarantee

Data leakage (using future information to predict past events) is the most common failure mode in sports prediction ML.

This system has a **single, centralised leakage guard** in `build_feature_row()`:

```python
_cutoff = pd.Timestamp(as_of_date)
matches = matches_df[matches_df["date"] < _cutoff].copy()
goalscorers = goalscorers_df[goalscorers_df["date"] < _cutoff].copy()
shootouts = shootouts_df[shootouts_df["date"] < _cutoff].copy()
substitutions = substitutions_df[substitutions_df["date"] < _cutoff].copy()
```

**All feature computation happens on data strictly before `as_of_date`.**

During training, `as_of_date = the date of the match being predicted`. During inference, `as_of_date = today`.

Rankings use snapshot-based data (already timestamped) so they are inherently safe.

#### Symmetric Augmentation & Leakage
Both team orientations (`A vs B` and `B vs A`) of the same match are generated. Both get the **same `as_of_date`**, so both look at history before the match. No cross-contamination occurs. In the validation split, both orientations of the same match stay in the **same fold** to prevent an indirect leakage path.

---

## 6. Training Pipeline

### 6.1 Data Filtering & Match Selection

Training uses **competitive matches only** — friendlies are excluded by default:

```python
competitive_keywords = [
    "world cup", "copa", "euro", "afcon", "asian cup",
    "qualifier", "qualification", "gold cup", "nations league", "confederation",
]
```

Optionally, **curated 2026 pre-World Cup friendlies** can be included via `ML_PRJCT_INCLUDE_FRIENDLIES_TRAIN=1`.

Training data starts from `TRAIN_FROM_YEAR = 2006` in standard mode. When historical-weighted mode is active (`USE_HISTORICAL_WEIGHTED = True`), data extends back to 1872 with older data heavily downweighted.

---

### 6.2 Sample Weighting

Each training row is assigned a **sample weight** that tells CatBoost how much to emphasise it:

```
sample_weight = era_weight(match_date) × tournament_weight(tournament)
```

#### Era Weights

| Era | Weight | Rationale |
|-----|--------|-----------|
| 2018–present | 1.25 | Most relevant to modern football |
| 2006–2017 | 1.00 | Baseline modern era |
| 2002–2005 | 0.45 | Some relevance — partial transition era |
| Pre-2002 | 0.08 | Very old data; minimal influence |

#### Tournament Weights

See Section 11 for the full tournament weight table. In brief: World Cup = 1.0, major continental tournaments = 0.85, qualifiers = 0.70, friendlies = 0.30.

---

### 6.3 Data Augmentation — Symmetric Team Swap

For every match, **two training rows are generated**: one with `team_a = home`, one with `team_a = away`:

```python
feat_a, _ = build_feature_row(team_a=home, team_b=away, ...)
label_a = _label(row, home)  # 0/1/2

feat_b, _ = build_feature_row(team_a=away, team_b=home, ...)
label_b = 2 - label_a  # Inverted
```

This doubles the training set and enforces that the model is **symmetric** — it should produce consistent predictions regardless of which team is listed first.

---

### 6.4 Time-Based Validation Split

The validation set is the **2022 FIFA World Cup** (held out completely during training and threshold calibration):

```
Train: all competitive matches from 2006 (or 1872) up to, but not including, 2022
Test:  all 2022 FIFA World Cup matches
```

This mimics real-world deployment — the model is evaluated on a complete future tournament it never saw during training.

The best model configuration is selected based on:
```
primary metric: macro F1
tiebreaker 1:  accuracy
tiebreaker 2:  draw F1
```

After validation, the **final model is retrained on the full dataset** (train + test) to maximise data before deployment.

---

### 6.5 Historical Source Capping

When using the historical (pre-2006) dataset, the system caps the non-World-Cup rows to avoid the older, lower-quality data overwhelming the training:

```
HISTORICAL_SOURCE_CAP = 5000
```

World Cup rows are always preserved in full. Non-WC rows are sampled:
- 75% from 2006+ matches
- 25% from pre-2006 matches

---

## 7. The Machine Learning Models

### 7.1 Why CatBoost?

CatBoost was chosen because:
1. **Native categorical feature support** — team names and tournaments are high-cardinality categoricals. CatBoost handles these natively via Ordered Target Statistics encoding, without requiring manual one-hot encoding.
2. **Ordered boosting** — CatBoost's Ordered mode prevents the target leakage common in naive gradient boosting implementations.
3. **Strong out-of-box performance** on tabular data with mixed types.
4. **Robustness** — handles missing values and outliers gracefully.
5. **Fixed algorithm** — the project is locked to CatBoost so that improvement effort focuses on better features and data rather than algorithm selection.

---

### 7.2 Result Classifier — Two-Stage Architecture

The result classifier uses a **Two-Stage CatBoost** architecture:

#### Stage 1: Draw vs Decisive Binary Classifier
```
Input:  All 153 features
Output: P(draw)  — probability that the match ends in a draw
Config: loss_function="Logloss", class_weights=[1.0, 2.5]
        (draw class weighted 2.5× to counteract draw scarcity)
```

#### Stage 2: Win vs Loss Binary Classifier (conditioned on decisive)
```
Input:  All 153 features (for non-draw training rows only)
Output: P(win | decisive) — conditional on the match not being a draw
Config: loss_function="Logloss", class_weights=[1.0, 1.0]
```

#### Probability Assembly
```python
P(loss) = P(decisive) × (1 − P(win | decisive))
P(draw) = P(draw) from Stage 1
P(win)  = P(decisive) × P(win | decisive)
P(decisive) = 1 − P(draw)
```

The three probabilities are normalised to sum to 1.0.

#### Why Two-Stage?

Draws are inherently hard to predict — only ~25% of matches end in draws. A standard multiclass classifier tends to be biased towards wins/losses. Separating the draw decision into its own binary classifier allows the model to be more precisely tuned to draw patterns.

The two-stage model is **competed against a standard multiclass CatBoost** on the validation set, and the one with better macro F1 is selected.

---

### 7.3 Draw Calibration

The draw prediction threshold is **calibrated** rather than fixed:

1. A separate calibration model is trained on all data before the previous World Cup (e.g., pre-2018 data for calibration year 2018)
2. That calibration model predicts draw probabilities for the previous WC matches
3. A threshold sweep from 0.08 to 0.45 (step 0.01) finds the threshold that maximises:
   - `macro_F1` (primary) → `draw_F1` (secondary) → `accuracy` (tertiary)
   - Subject to: overall accuracy ≥ 55% guard

The calibrated threshold is then used for both validation and final model deployment.

**Default draw threshold:** 0.24 (if calibration yields no better option)

---

### 7.4 Goal Prediction Model

A **separate CatBoost Regressor** predicts the number of goals for each team.

**Architecture:**
- Two independent `CatBoostRegressor` models: one predicts `team_a_goals`, one predicts `team_b_goals`
- `loss_function = "RMSE"` (Root Mean Square Error)
- 600 boosting iterations, depth 6, learning rate 0.04, L2 regularisation = 6

**Features for goal model** (a simplified subset of the result model features):

| Category | Features |
|---------|---------|
| Venue | `team_a_is_home`, `neutral_venue`, `team_a_host_country`, etc. |
| Tournament | `tournament_weight` (numeric), tournament category (categorical) |
| Team form | `recent_goals_for/against`, `win_rate`, `draw_rate`, `clean_sheet_rate`, `over_2_5_rate` |
| Home/Away split | `home_goals_for/against`, `away_goals_for/against` |
| H2H | `h2h_team_a_goals_for`, `h2h_team_b_goals_for`, `h2h_over_2_5_rate` |
| Rankings | `team_a_rank`, `team_b_rank`, `rank_diff`, `rank_points` |

**Symmetry averaging during inference** (same as result model):
```python
pred_a = (model_a.predict(AB_row) + model_b.predict(BA_row)) / 2
pred_b = (model_b.predict(AB_row) + model_a.predict(BA_row)) / 2
```

**Scoreline derivation via Poisson distribution:**
```python
# For expected_a goals λ_a and expected_b goals λ_b:
for a in range(8):  # 0 to 7 goals
    for b in range(8):
        P(a, b) = Poisson(λ_a, a) × Poisson(λ_b, b)

most_likely_scoreline = argmax P(a, b)
over_2_5_probability = sum(P(a, b) for a + b > 2.5)
```

---

## 8. Inference Pipeline (Prediction)

When `predict_match(team_a, team_b, ...)` is called:

### 8.1 Symmetry Averaging

Two feature rows are built:
- **Direct:** `build_feature_row(team_a, team_b, venue_context=normal)`
- **Mirrored:** `build_feature_row(team_b, team_a, venue_context=flipped)`

Both rows are passed through the model separately. The final probabilities are:

```python
P(loss)  = (direct_P(loss)  + mirrored_P(win))  / 2
P(draw)  = (direct_P(draw)  + mirrored_P(draw))  / 2
P(win)   = (direct_P(win)   + mirrored_P(loss))  / 2
```

This averaging enforces **symmetry consistency** — the prediction for `A vs B` is guaranteed to be the logical inverse of `B vs A`.

---

### 8.2 2026 World Cup Context Adjustment

If WC 2026 squad and team data is available, the base model probabilities are **adjusted post-hoc** based on squad strength signals not captured in the historical data:

**Squad strength score** is computed per team from four components:
```
squad_strength = rank_score × 0.52
               + history_score × 0.22
               + key_player_score × 0.18
               + confederation_score × 0.08
               − debut_penalty (0.06 for WC debut teams)
```

**Adjustment formula:**
```python
delta = team_a.strength − team_b.strength
depth_delta = team_a.squad_depth − team_b.squad_depth
age_delta = team_a.age_balance − team_b.age_balance
top_league_delta = team_a.top_league_share − team_b.top_league_share

combined_delta = (delta × 0.72) + (depth_delta × 0.18)
              + (age_delta × 0.05) + (top_league_delta × 0.05)

win_shift = clip(combined_delta × 0.11, −0.085, +0.085)
draw_shift = ((closeness − 0.5) × 0.025) − (|win_shift| × 0.20)
```

The adjusted probabilities are re-normalised to sum to 1.0. The maximum possible shift from this adjustment is ±8.5 percentage points.

---

### 8.3 Label Decision Policy

The result classifier first produces the probability distribution and a
classifier-only label:

**Two-stage model:**
```python
if P(draw) >= draw_threshold:
    classifier_label = Draw
elif P(win) >= P(loss):
    classifier_label = Win
else:
    classifier_label = Loss
```

**Multiclass model:**
```python
if P(draw) >= draw_threshold AND (max(P(win), P(loss)) − P(draw)) <= 0.15:
    classifier_label = Draw
else:
    classifier_label = argmax(P(loss), P(draw), P(win))
```

The `draw_threshold` is the calibrated value from Section 7.3 (default: 0.24).

The goal model also produces an aggregate outcome distribution from its
Poisson scoreline grid:

```python
P_goal(Team A win) = sum P(a, b) where a > b
P_goal(Draw)       = sum P(a, b) where a == b
P_goal(Team B win) = sum P(a, b) where b > a
```

The public predicted outcome is resolved from a calibrated blend of the
classifier distribution and the goal distribution:

```python
P_final = (classifier_weight * P_classifier) + (goal_weight * P_goal)
predicted = argmax(P_final(Team B win), P_final(Draw), P_final(Team A win))
```

The blend weights are stored in `model/artifacts/outcome_policy.json` and can be
recalibrated with `scripts/calibrate_outcome_policy.py`. After the final label is
chosen, the displayed scoreline is selected from the goal model's best scoreline
for that same outcome. A predicted draw therefore displays a draw scoreline, a
Team A win displays a Team A winning scoreline, and a Team B win displays a Team
B winning scoreline.

If the goal model is unavailable, the system falls back to the classifier-only
label above. The classifier probabilities remain visible as supporting context.

---

### 8.4 Confidence Score

```python
base = max(P(win), P(draw), P(loss))
data_coverage = active_feature_groups / total_feature_groups
h2h_bonus = min(h2h_meetings / 20, 0.10)

confidence = (base × 0.70) + (data_coverage × 0.20) + h2h_bonus
confidence = clip(confidence, 0.0, 1.0)
```

| Label | Threshold |
|-------|-----------|
| Very High | > 0.80 |
| High | > 0.65 |
| Medium | > 0.50 |
| Low | ≤ 0.50 |

**Active feature groups** (out of 9 total):
1. H2H + Form (matches.csv)
2. Tournament stats (shootouts.csv)
3. FIFA Rankings (rankings.csv)
4. Goalscoring (goalscorers.csv)
5. Super-sub (goalscorers + substitutions)
6. Copa America 2024 data
7. International Friendlies 2026 data
8. EURO 2024 player data
9. Player appearances/goals (player_appearances.csv, player_goals.csv)

The confidence score rewards predictions backed by more complete data sources. A match between two unknown teams with no historical data would receive a Low confidence rating even if the model's probability distribution is decisive.

---

## 9. All Parameters & Configuration

All tuneable parameters can be overridden via **environment variables**.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ML_PRJCT_DRAW_CLASS_WEIGHT` | `1.6` | Class weight for draw in multiclass model |
| `ML_PRJCT_DRAW_PROB_THRESHOLD` | `0.24` | Default draw probability threshold |
| `ML_PRJCT_DRAW_CLOSE_MARGIN` | `0.15` | Max margin for draw override in multiclass |
| `ML_PRJCT_USE_TWO_STAGE_RESULT` | `1` (on) | Enable two-stage result model |
| `ML_PRJCT_DRAW_BINARY_WEIGHT` | `2.5` | Draw class weight in binary stage-1 model |
| `ML_PRJCT_DRAW_THRESHOLD_MIN` | `0.08` | Lower bound for draw threshold sweep |
| `ML_PRJCT_DRAW_THRESHOLD_MAX` | `0.45` | Upper bound for draw threshold sweep |
| `ML_PRJCT_DRAW_THRESHOLD_STEP` | `0.01` | Step size for threshold sweep |
| `ML_PRJCT_DRAW_CALIBRATION_MIN_ACCURACY` | `0.55` | Min accuracy guard for threshold calibration |
| `ML_PRJCT_USE_HISTORICAL_WEIGHTED` | `1` (on) | Use weighted pre-2006 historical data |
| `ML_PRJCT_HISTORICAL_SOURCE_CAP` | `5000` | Max non-WC rows in historical dataset |
| `ML_PRJCT_HISTORICAL_FAST_FEATURES` | `1` (on) | Skip slow features for historical training |
| `ML_PRJCT_INCLUDE_COPA` | `1` (on) | Include Copa America context features |
| `ML_PRJCT_INCLUDE_FRIENDLIES` | `1` (on) | Include 2026 friendlies context features |
| `ML_PRJCT_INCLUDE_FRIENDLIES_TRAIN` | `0` (off) | Include friendlies matches in training set |
| `ML_PRJCT_TRAIN_SAMPLE_SIZE` | *(unset)* | Cap training rows for quick experiments |

**Hardcoded constants** (change in source code):

| Constant | Value | Location | Description |
|---------|-------|----------|-------------|
| `TRAIN_FROM_YEAR` | 2006 | `train.py` | Earliest year in standard training mode |
| `VALIDATION_YEAR` | 2022 | `train.py` | WC held out for model evaluation |
| `CURATED_FRIENDLIES_FROM_YEAR` | 2026 | `train.py` | Friendlies included from this year |
| `RANDOM_STATE` | 42 | `train.py` | Reproducibility seed |
| `DATA_FROM_YEAR` | 2006 | `features.py` | Earliest year for player stats |
| `SUPERSUB_MIN_APPEARANCES` | 3 | `supersub.py` | Min sub apps to qualify |
| `SUPERSUB_MIN_RATE` | 0.40 | `supersub.py` | Min goal-as-sub rate |
| `SUB_ENTRY_MINUTE_PROXY` | 60 | `supersub.py` | Minute after which goals = likely sub |

---

## 10. Full Feature Column Reference

### Categorical (3 columns)
```
team_a, team_b, tournament
```

### Numeric (150 columns)

**Venue Context (5)**
```
team_a_is_home, team_b_is_home, neutral_venue,
team_a_host_country, team_b_host_country
```

**H2H (15)**
```
h2h_total_meetings, h2h_team_a_wins, h2h_team_b_wins, h2h_draws,
h2h_wc_only_team_a_wins, h2h_wc_only_team_b_wins,
h2h_last5_team_a_wins, h2h_last5_goals_a, h2h_last5_goals_b,
h2h_weighted_win_ratio_a
```

**Team A Form (8)**
```
team_a_form_wins, team_a_form_draws, team_a_form_losses,
team_a_goals_scored_avg, team_a_goals_conceded_avg,
team_a_clean_sheet_rate, team_a_win_streak, team_a_unbeaten_streak
```

**Team B Form (8)** — same with `team_b_` prefix

**Tournament History per team (5 × 2 = 10)**
```
team_a_wc_titles, team_a_wc_finals_reached, team_a_wc_win_rate_knockouts,
team_a_penalty_shootout_wins, team_a_penalty_shootout_losses
```

**Rankings (7)**
```
team_a_rank, team_b_rank, rank_diff, rank_abs_diff, rank_closeness,
team_a_rank_points, team_b_rank_points
```

**Draw Tendency / Team Similarity (9)**
```
h2h_draw_rate, h2h_balance_score, combined_form_draw_rate,
form_draw_rate_diff, attack_balance_abs_diff, defense_balance_abs_diff,
combined_clean_sheet_rate, low_scoring_tendency, draw_similarity_score
```

**Goalscoring (4)**
```
team_a_avg_goals_wc, team_b_avg_goals_wc,
team_a_scoring_first_win_rate, team_b_scoring_first_win_rate
```

**Super-Sub (2)**
```
team_a_has_supersub, team_b_has_supersub
```

**Copa America 2024 (20)**
```
team_a_copa_pre_match_ppg, team_b_copa_pre_match_ppg, ...
copa_league_avg_goals, copa_league_xg_avg, copa_league_home_advantage
```

**International Friendlies 2026 (20)** — same structure as Copa

**EURO 2024 (22 × 2 = 44)**
```
team_a_euro_squad_impact, team_a_euro_top5_impact,
team_a_euro_attack_impact, team_a_euro_midfield_impact, ...
```

**Player Impact (6 × 2 = 12)**
```
team_a_top_scorer_wc_goals, team_a_attack_rating,
team_a_golden_ball_count, team_a_golden_boot_count,
team_a_golden_glove_count, team_a_avg_player_experience
```

---

## 11. Tournament Weights

Used in H2H weighting, sample weight computation, and goal model features.

| Tournament | Weight |
|-----------|--------|
| FIFA World Cup | **1.00** |
| Copa América | 0.85 |
| UEFA Euro | 0.85 |
| Africa Cup of Nations (AFCON) | 0.85 |
| AFC Asian Cup | 0.85 |
| CONCACAF Gold Cup | 0.75 |
| FIFA Confederations Cup | 0.75 |
| World Cup Qualification | 0.70 |
| UEFA Nations League | 0.65 |
| International Friendly | 0.30 |
| Unknown/other | 0.50 |

---

## 12. Model Evaluation Metrics

The following metrics are computed on the held-out 2022 World Cup validation set:

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | Overall fraction of correctly predicted outcomes |
| **Balanced Accuracy** | Accuracy averaged per class — guards against class imbalance |
| **Log Loss** | Probability calibration quality (lower = better-calibrated probabilities) |
| **Macro F1** | F1 score averaged equally across Win/Draw/Loss classes (primary selection metric) |
| **Weighted F1** | F1 weighted by class frequency |
| **Draw Precision** | Of predicted draws, how many were actual draws? |
| **Draw Recall** | Of actual draws, how many were predicted correctly? |
| **Draw F1** | Harmonic mean of draw precision and recall |

**Why macro F1 as primary?**
Macro F1 treats each class equally, regardless of frequency. Since draws are less common (~25% of matches), accuracy alone would reward a model that never predicts draws. Macro F1 forces the model to perform well on all three outcomes.

---

## 13. Known Limitations & Design Decisions

### 1. No lineup data
The model does not know which players will be on the pitch. It uses squad-level indicators (super-sub flags, average player experience, attack ratings) but cannot account for individual absences, suspensions, or tactical lineup changes. This is the single biggest source of uncertainty.

### 2. No in-game state
The model predicts from scratch — it cannot be updated mid-game. It is designed for pre-match prediction only.

### 3. World Cup stage column missing
The `matches.csv` dataset does not include a match stage column (e.g., Group Stage, Quarter-Final). WC titles are stored in a hardcoded dictionary. Knockout win rate is a proxy for stage-specific performance, not exact knockout data.

### 4. Draw prediction is inherently difficult
Draws are the most difficult outcome to predict in football — the model explicitly counteracts this with:
- Higher class weights for draw in training
- A dedicated binary draw classifier (two-stage)
- Threshold calibration on historical WC data
- Composite draw-tendency features

Despite this, draw F1 is typically lower than Win/Loss F1.

### 5. Bayesian smoothing for small samples
Several ratio features (scoring-first win rate, H2H ratios for rarely-met teams) use Laplace smoothing to prevent extreme values from one or two matches dominating predictions.

### 6. Post-hoc 2026 squad adjustment is capped
The squad strength adjustment (Section 8.2) is capped at ±8.5 percentage points. This is intentional — the historical model is the primary predictor and the squad context only provides a gentle nudge, not an override.

### 7. Confidence score is an honest impurity measure
The confidence score deliberately penalises predictions for teams with little historical data. A prediction for a WC debut team with no ranking history will receive a Low confidence label even if the model outputs decisive probabilities — reflecting genuine uncertainty about data coverage.

---

*Documentation written for the FIFA World Cup 2026 Match Prediction System.*
*Source files: [`model/train.py`](model/train.py) · [`model/predict.py`](model/predict.py) · [`model/features.py`](model/features.py) · [`data/preprocess.py`](data/preprocess.py) · [`model/goals.py`](model/goals.py)*
