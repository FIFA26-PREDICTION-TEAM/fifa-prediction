# What I Learned While Building My FIFA World Cup 2026 Prediction Project

**Student:** Suhayl Ahmed  
**Project:** FIFA World Cup 2026 Match Prediction App  
**Report type:** Learning reflection and model explanation  
**Last updated:** June 21, 2026

## 1. Introduction

This report focuses on the parts of the project that I personally worked on and learned from. The project is a FIFA World Cup 2026 prediction app, but my main learning was about understanding the machine learning workflow: training a model, evaluating it, improving the final prediction policy, testing changes, debugging issues, and explaining the work clearly.

Some parts of the wider project were handled by my friend, so I am not taking credit for those parts. This report focuses on what I learned and what is happening inside the model.

## 2. What The Project Does

The application predicts international football match outcomes. A user chooses two teams, and the system gives prediction information such as:

- Win probability
- Draw probability
- Loss probability
- Expected goals for each team
- Rounded expected-goals scoreline
- Confidence in the prediction
- A final predicted outcome

My focus was understanding how the model produces these predictions and how to measure whether the model is performing well.

## 3. Main Things I Learned

| Area | What I Learned |
| --- | --- |
| Machine learning | How to train, evaluate, compare, and improve models. |
| Classification | How a model predicts categories such as win, draw, or loss. |
| Regression | How a model predicts numbers such as expected goals. |
| F1 score | Why accuracy alone is not enough for evaluating a model. |
| Draw prediction | Why some football outcomes are harder to predict than others. |
| Calibration | How the final decision rule can change model performance. |
| Testing | How tests confirm that changes did not break prediction behavior. |
| Debugging | How to use logs, metrics, and repeated checks to understand problems. |
| Documentation | How to explain technical work clearly and honestly. |

## 4. What Is Going On Inside The Model

The project uses two main machine learning ideas:

| Prediction Task | Type Of Model | Output |
| --- | --- | --- |
| Match result | Classification model | Team A loss, draw, or Team A win |
| Expected goals | Regression model | Number of goals expected for each team |

The result model answers a classification question:

> Which category is most likely: Team A win, draw, or Team B win?

The expected-goals model answers a regression question:

> How many goals is each team expected to score?

I learned that these are different machine learning problems. A classifier is used when the output is a category. A regressor is used when the output is a number.

## 5. Why The Model Uses Probabilities

The model does not only say "Team A will win." It gives probabilities for each possible outcome.

Example:

| Outcome | Meaning |
| --- | --- |
| Team A win probability | How likely Team A is to win. |
| Draw probability | How likely the match is to end in a draw. |
| Team B win probability | How likely Team B is to win. |

This helped me learn that a good prediction system should show uncertainty. Football is unpredictable, so probabilities are more useful than a single yes-or-no answer.

## 6. What I Learned About Model Training

Training means teaching the model from previous examples. The model studies past matches and learns patterns that are useful for future predictions.

I learned that training is not just running a model once. It also includes:

1. Choosing the prediction target.
2. Training the model on examples.
3. Testing it on matches it has not seen during training.
4. Measuring the performance.
5. Comparing different model versions.
6. Keeping the version that performs better on validation results.

The most important lesson was that model changes should be judged by evidence, not by guessing.

## 7. What I Learned About F1 Score

One of the most important things I learned was that accuracy alone can be misleading.

In football prediction, there are three possible classes:

| Class | Meaning |
| --- | --- |
| Loss | Team A loses. |
| Draw | Both teams finish level. |
| Win | Team A wins. |

If the model predicts wins and losses well but performs badly on draws, the accuracy might still look acceptable. F1 score helps show whether the model is performing well across all classes.

| Metric | What I Learned |
| --- | --- |
| Accuracy | Shows total correct predictions, but can hide weak performance on one class. |
| Macro F1 | Treats each class more equally. |
| Weighted F1 | Gives more weight to classes that appear more often. |
| Balanced accuracy | Helps compare performance across classes. |
| Log loss | Measures how good the probability estimates are. |

## 8. Final Model Evaluation

The final classifier was evaluated on a World Cup validation set.

| Metric | Score |
| --- | ---: |
| Accuracy | 0.6094 |
| Macro F1 | 0.4918 |
| Weighted F1 | 0.5831 |
| Balanced accuracy | 0.5025 |

I learned how to compare model versions using the same validation setup. The previous macro F1 was 0.4823, and after improving the training setup and final prediction calibration, the macro F1 became 0.4918.

This taught me that even a small improvement is meaningful when it is measured consistently.

## 9. What I Learned About Draw Prediction

Draw prediction was one of the hardest parts of the project. A draw is difficult because many matches are close to being either a draw or a narrow win.

I learned that the model needs special handling for draws because:

- Draws are not always as common as wins or losses.
- A draw can be close to both possible win outcomes.
- The raw highest probability may not always be the best final decision.
- A threshold can help decide when a draw should be selected.

| Draw Topic | What I Learned |
| --- | --- |
| Class imbalance | Some outcomes happen less often, so the model may ignore them. |
| Draw threshold | A separate rule can help decide when a draw is likely enough. |
| Close matches | Draws are more reasonable when the teams are close in predicted strength. |
| Validation tuning | The best rule should be selected using validation results. |

This helped me understand that model performance can depend not only on the model itself, but also on the final decision rule.

## 10. What I Learned About Expected Goals

I learned that predicting the exact final score is very difficult. Instead of only predicting one score, the project predicts expected goals.

Expected goals are useful because they estimate how many goals each team may score on average.

Example:

| Expected Goals | Rounded Display |
| --- | --- |
| Team A xG = 2.06 | 2 |
| Team B xG = 0.77 | 1 |
| Rounded score | 2-1 |

This taught me that model outputs sometimes need to be converted into a simpler format so people can understand them easily.

## 11. Expected-Goals Evaluation

The expected-goals model was evaluated using error metrics.

| Metric | Score |
| --- | ---: |
| Combined MAE | 0.9332 |
| Team A MAE | 0.9305 |
| Team B MAE | 0.9360 |
| Exact score accuracy | 0.1140 |
| Over 2.5 goals accuracy | 0.6158 |

I learned that exact score prediction is naturally hard. The exact score accuracy shows that even when expected goals are useful, predicting the exact final score is much more difficult.

## 12. What I Learned About Calibration

I learned that the model's raw probabilities are not always the complete final answer. The system still needs a final policy to decide what result should be displayed.

The final prediction policy was recalibrated using validation results.

| Policy Version | Macro F1 |
| --- | ---: |
| Previous final policy | 0.4463 |
| New final policy | 0.4918 |

The best final policy used the classifier decision label directly because it performed better than blending the classifier and expected-goals outputs.

This taught me an important lesson:

> A simpler final decision rule can sometimes perform better than a more complicated one.

## 13. What I Learned About Testing

Testing helped me confirm that model and prediction changes worked correctly. I learned that even if the app seems to run, the prediction logic still needs to be checked.

The tests checked things such as:

| Test Area | What It Checks |
| --- | --- |
| Prediction decision logic | The final predicted label is consistent with the probabilities. |
| Scoreline behavior | The rounded scoreline matches the predicted outcome. |
| Probability validity | Win, draw, and loss probabilities stay between 0 and 1 and add up correctly. |
| Team-order symmetry | Reversing the teams gives a consistent opposite prediction. |
| Goal prediction behavior | Expected-goals outputs remain valid and symmetric. |

Latest test result:

| Result | Count |
| --- | ---: |
| Passed tests | 23 |
| Failed tests | 0 |
| Warnings | 3 |

This taught me that testing gives confidence when improving the model and prediction logic.

## 14. What I Learned About Debugging

While working on the project, I learned that debugging is a normal part of development. Some issues had to be solved by checking outputs carefully, comparing metrics, and running focused tests.

| Debugging Task | What I Learned |
| --- | --- |
| Checking logs | Logs help show what stage a process is running. |
| Reading metrics | Metrics show whether a change actually improved the model. |
| Comparing model versions | A change should only be kept if the measured result improves. |
| Running focused tests | Smaller checks help find the exact failing area. |
| Cleaning generated outputs | Extra generated outputs should not be confused with intentional work. |

I learned that debugging is not only about fixing errors. It is also about proving that a change helped.

## 15. Problems I Worked On

| Problem | How I Worked On It | What I Learned |
| --- | --- | --- |
| Needed to understand model quality | Reviewed evaluation metrics and F1 score. | Metrics are essential for judging a model. |
| Accuracy was not enough | Focused on macro F1 and weighted F1. | Different metrics explain different parts of performance. |
| Draw predictions were difficult | Used threshold and validation policy tuning. | Some outcomes need special handling. |
| Final prediction policy needed improvement | Compared policy results and kept the better one. | Final decision rules matter. |
| Changes needed safety checks | Ran automated tests. | Testing protects the project from regressions. |
| Report needed to explain learning | Wrote this learning-focused report. | Documentation should be clear and honest. |

## 16. Skills I Practiced

| Skill | Example From The Project |
| --- | --- |
| Understanding model behavior | I studied how the classifier and expected-goals model work. |
| Training models | I trained and compared machine learning models. |
| Evaluating models | I reviewed F1 score, accuracy, log loss, MAE, and RMSE. |
| Improving model usage | I compared configurations and kept the better validation result. |
| Running tests | I checked prediction behavior using automated tests. |
| Debugging | I used logs and metrics to understand what happened. |
| Writing documentation | I wrote this report to explain what I learned. |

## 17. Most Important Lessons

1. Machine learning projects require careful evaluation, not only training.
2. Accuracy alone can be misleading.
3. F1 score is useful when classes are imbalanced.
4. Draw prediction is harder than normal win-or-loss prediction.
5. Expected goals and match outcome are related but different tasks.
6. A final prediction policy can affect performance.
7. Automated tests help keep a project reliable.
8. Metrics are important evidence.
9. Debugging requires patience and careful checking.
10. Documentation should explain both the work and the learning.

## 18. Challenges I Faced

| Challenge | What I Learned From It |
| --- | --- |
| Understanding model metrics | I learned the difference between accuracy, F1, and log loss. |
| Improving F1 score | I learned that model performance improves through careful experiments. |
| Handling draw prediction | I learned that some outcomes are harder to predict than others. |
| Checking if changes worked | I learned to rely on validation metrics and tests. |
| Explaining the project | I learned how to write a clear report. |

## 19. Final Reflection

This project helped me understand the machine learning workflow more clearly. I learned how a prediction model is trained, how it is evaluated, how F1 score is used, how expected goals are different from match outcome prediction, how final prediction calibration works, and why testing is important.

The most important thing I learned is that a machine learning result should always be supported by evidence. In this project, that evidence came from validation metrics, F1 score, expected-goals error metrics, and automated tests.

By the end of the project, I had a much better understanding of how model training, evaluation, testing, debugging, and documentation fit together in a real application.
