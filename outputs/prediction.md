# Breakthrough prediction

Definition: An artist breaks through if, in the target year (the observation year plus 3 years), they are in the upper right quadrant, both centered coordinates greater than zero, or in the top quartile of centered streams, having started below that at the observation year. Artists already in that region at observation are excluded, since the axes are centered on the yearly median the origin is the median.

Observation: first full year on the plane, movement over the next year. Horizon: 3 years.

Eligible artists: 130 (91 train, 39 test). Base rate of breakthrough: 0.192.

## Held out performance vs baselines

| Model | Accuracy | ROC AUC | Precision | Recall | F1 |
|-------|---------:|--------:|----------:|-------:|---:|
| Logistic regression | 0.821 | 0.879 | 0.545 | 0.750 | 0.632 |
| Coin toss | 0.500 | 0.500 | . | . | . |
| Majority class | 0.795 | 0.500 | . | . | . |

Confusion matrix on test: true negatives 26, false positives 5, false negatives 2, true positives 6.

## Sample predictions

| Case | Artist | Early placements | Early streams | Predicted probability | Prediction | Actual |
|------|--------|-----------------:|--------------:|----------------------:|:----------:|:------:|
| TP | Pool Artist 0082 | -0.15 | -0.40 | 0.56 | breakthrough | breakthrough |
| TP | Pool Artist 0196 | 0.10 | -0.32 | 0.77 | breakthrough | breakthrough |
| TN | Pool Artist 0101 | -0.50 | -0.14 | 0.24 | no | no |
| TN | Pool Artist 0088 | -0.77 | -0.80 | 0.07 | no | no |
| FP | Pool Artist 0116 | -0.03 | 0.19 | 0.65 | breakthrough | no |
| FP | Pool Artist 0054 | 0.08 | -0.09 | 0.73 | breakthrough | no |
| FN | Pool Artist 0093 | -0.27 | -0.43 | 0.38 | no | breakthrough |
| FN | Pool Artist 0274 | -0.19 | -0.45 | 0.46 | no | breakthrough |

Cases: TP true positive, TN true negative, FP false positive, FN false negative.
