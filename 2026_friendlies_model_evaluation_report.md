# 2026 Pre-World-Cup Friendlies Model Evaluation Report

## Executive Summary

The saved football match prediction model was evaluated on the 2026 pre-World-Cup international friendlies dataset:

```text
data/external/fifa_pre_world_cup_friendlies_2026.csv
```

The main match-level results were:

| Metric | Score |
|---|---:|
| Accuracy | 65.52% |
| Macro F1 | 0.5580 |
| Weighted F1 | 0.6581 |
| Balanced Accuracy | 0.5660 |
| Log Loss | 0.6647 |

The model correctly predicted 38 out of 58 matches.

```text
38 / 58 = 0.6552 = 65.52%
```

The macro F1 score was lower than accuracy because the test set was imbalanced and the model performed better on the majority class, Team1 wins, than on draws and Team1 losses.

## Dataset Used

The evaluation used the 2026 pre-World-Cup friendly results file:

```text
data/external/fifa_pre_world_cup_friendlies_2026.csv
```

The test file contained 58 completed matches.

Date range:

```text
2026-05-22 to 2026-06-07
```

Actual result distribution:

| Result Type | Count |
|---|---:|
| Team1 win | 39 |
| Draw | 11 |
| Team2 win | 8 |

This distribution matters because Team1 wins were the dominant class. A model can therefore achieve decent accuracy by doing well on Team1 wins, even if it is weaker on draws or Team1 losses.

## Prediction Task

Each match was evaluated as a three-class classification problem.

The model did not predict exact scores. It predicted the match outcome class:

| Label | Meaning |
|---:|---|
| 0 | Team1 loss |
| 1 | Draw |
| 2 | Team1 win |

For example, if the CSV row was:

```text
Argentina vs Honduras, Argentina 2-0 Honduras
```

then the correct class was:

```text
2 = Team1 win
```

If the row was:

```text
Morocco vs Norway, Morocco 1-1 Norway
```

then the correct class was:

```text
1 = Draw
```

## Model Evaluated

The evaluation used the saved model artifact:

```text
model/artifacts/model.pkl
```

The metadata file:

```text
model/artifacts/meta.json
```

reported the saved model as:

| Property | Value |
|---|---|
| Model name | TwoStageCatBoost |
| Result model type | two_stage |
| Draw threshold | 0.37 |
| Validation target | world_cup_2022 |

The model is a CatBoost-based classifier. CatBoost is suitable for this project because the data is tabular and includes categorical signals such as teams, tournaments, venues, and match context.

## Two-Stage Prediction Logic

The saved model uses a two-stage result prediction setup.

Stage 1 predicts whether the match is likely to be a draw.

Stage 2 predicts the winner if the match is not classified as a draw.

The final decision works like this:

1. The draw model estimates the probability of a draw.
2. If the draw probability is at least the calibrated draw threshold, the final prediction becomes draw.
3. Otherwise, the decisive model chooses between Team1 win and Team1 loss.

For this saved artifact, the draw threshold was:

```text
0.37
```

That means a match was predicted as a draw when the model's draw probability was high enough to cross that threshold.

## Feature Construction

The test matches were converted into model-ready feature rows using the project's normal feature-building pipeline:

```text
build_training_data(...)
```

The features included signals such as:

| Feature Group | Examples |
|---|---|
| Venue context | neutral venue, home/away indicators |
| Head-to-head history | prior meetings, wins, draws, weighted win ratio |
| Recent form | recent wins, draws, losses, goals scored, goals conceded |
| Tournament history | World Cup titles, World Cup record, shootout history |
| FIFA rankings | rank, ranking points, rank difference |
| Draw tendency | form draw rate, H2H draw rate, similarity score |
| Context features | Copa America, EURO 2024, and friendly data where available |

The evaluation removed the 58 target matches from the feature-history frame before scoring. This prevented the feature builder from using those same test results as prior match history.

## How Accuracy Was Calculated

Accuracy measures the percentage of matches where the predicted class matched the actual class.

Formula:

```text
accuracy = number of correct predictions / total number of predictions
```

The confusion matrix from the 58-match evaluation was:

```text
Labels: Team1 Loss, Draw, Team1 Win

[[ 4, 1, 3],
 [ 0, 5, 6],
 [ 6, 4, 29]]
```

This means:

| Actual Class | Correct Predictions | Total Actual |
|---|---:|---:|
| Team1 loss | 4 | 8 |
| Draw | 5 | 11 |
| Team1 win | 29 | 39 |

Correct predictions:

```text
4 + 5 + 29 = 38
```

Total matches:

```text
58
```

Accuracy:

```text
38 / 58 = 0.6552
```

Final accuracy:

```text
65.52%
```

## How Macro F1 Was Calculated

Macro F1 is the average of the F1 scores for each class.

It gives equal importance to each class:

```text
Team1 loss
Draw
Team1 win
```

This is different from accuracy. Accuracy counts all matches together, while macro F1 checks how well the model performed on each class separately and then averages the class-level F1 scores.

For each class:

```text
precision = true positives / predicted positives
recall = true positives / actual positives
F1 = 2 * precision * recall / (precision + recall)
```

Then:

```text
macro F1 = average(F1_Team1Loss, F1_Draw, F1_Team1Win)
```

The class-level results were:

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Team1 loss | 0.4000 | 0.5000 | 0.4444 | 8 |
| Draw | 0.5000 | 0.4545 | 0.4762 | 11 |
| Team1 win | 0.7632 | 0.7436 | 0.7532 | 39 |

Macro F1:

```text
(0.4444 + 0.4762 + 0.7532) / 3 = 0.5580
```

Final macro F1:

```text
0.5580
```

## Why Accuracy Was Higher Than Macro F1

Accuracy was 65.52%, but macro F1 was 0.5580.

This happened because the dataset was imbalanced:

| Class | Count |
|---|---:|
| Team1 win | 39 |
| Draw | 11 |
| Team1 loss | 8 |

The model performed strongest on Team1 wins:

```text
29 correct Team1 wins out of 39 actual Team1 wins
```

Because Team1 wins were the majority class, this helped accuracy a lot.

However, macro F1 treated each class equally. It penalized the model because performance on draws and Team1 losses was weaker:

```text
Team1 loss F1: 0.4444
Draw F1:      0.4762
Team1 win F1: 0.7532
```

So the macro F1 score gives a more balanced view of model quality across all result types.

## Draw Performance

Draw-specific metrics:

| Metric | Score |
|---|---:|
| Draw precision | 0.5000 |
| Draw recall | 0.4545 |
| Draw F1 | 0.4762 |
| Actual draws | 11 |
| Predicted draws | 10 |

Interpretation:

The model predicted 10 draws.

Of those predicted draws, 5 were correct.

```text
draw precision = 5 / 10 = 0.5000
```

There were 11 actual draws.

The model found 5 of them.

```text
draw recall = 5 / 11 = 0.4545
```

The draw F1 score combines draw precision and draw recall:

```text
draw F1 = 0.4762
```

This shows that the model had some ability to identify draws, but draw prediction remained one of the harder parts of the task.

## Full Metric Summary

| Metric | Value | Meaning |
|---|---:|---|
| Accuracy | 0.6552 | Correct predictions divided by total matches |
| Macro F1 | 0.5580 | Average F1 across Team1 loss, draw, and Team1 win |
| Weighted F1 | 0.6581 | F1 averaged by class frequency |
| Balanced Accuracy | 0.5660 | Average recall across classes |
| Log Loss | 0.6647 | Probability quality, lower is better |
| Draw Precision | 0.5000 | Correct draw predictions divided by predicted draws |
| Draw Recall | 0.4545 | Correct draw predictions divided by actual draws |
| Draw F1 | 0.4762 | Combined draw precision and draw recall |

## Important Caveat About Data Leakage

This evaluation used the current saved model artifact.

The saved metadata says:

```text
include_friendlies_train: true
data_end: 2026-06-07
```

The test file also ends on:

```text
2026-06-07
```

This means the saved model may already have been trained using 2026 friendlies, possibly including some or all of the matches used for this evaluation.

To reduce leakage during scoring, the 58 test matches were removed from the feature-history frame before building test features. That means the feature builder did not use those exact matches as previous match history.

However, because the trained CatBoost artifact itself may already have seen those match labels during training, this should not be treated as a fully clean holdout score.

The safest wording is:

```text
The current saved model artifact scored 65.52% accuracy and 0.5580 macro F1
on the 58-match 2026 pre-World-Cup friendlies file.
```

A stricter evaluation would retrain the model with those 58 friendlies explicitly excluded from training, then evaluate the newly trained model on the same 58 matches.

## Final Interpretation

The model achieved 65.52% accuracy because it correctly classified 38 out of 58 matches.

The model's strongest area was predicting Team1 wins, which were also the most common result in the test data.

The macro F1 score of 0.5580 shows that the model was less balanced across all three classes. It did reasonably well on Team1 wins but had weaker performance on draws and Team1 losses.

Therefore, the headline result is:

```text
Accuracy: 65.52%
Macro F1: 0.5580
```

But the more careful conclusion is:

```text
The saved model artifact performed moderately well on the 2026 pre-World-Cup
friendlies file, especially on Team1 wins, but the result should be interpreted
with caution because the saved artifact may have already included 2026 friendlies
in training.
```