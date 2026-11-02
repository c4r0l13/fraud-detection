# Results (test period: last 20 days)

|                         |   pr_auc |   roc_auc |   alerts_per_day |   precision |   recall |   dollar_recall |     cost |   savings_vs_no_screening |
|:------------------------|---------:|----------:|-----------------:|------------:|---------:|----------------:|---------:|--------------------------:|
| Rules engine (baseline) |  nan     |   nan     |            16.5  |       0.048 |    0.084 |           0.223 | 28600.9  |                   6066.26 |
| Logistic regression     |    0.378 |     0.996 |            36.25 |       0.232 |    0.88  |           0.901 |  7046.79 |                  27620.3  |
| LightGBM                |    0.922 |     0.998 |            37.5  |       0.251 |    0.984 |           0.987 |  4216.37 |                  30450.8  |

Cost with no screening: $34,667

## Recall by scenario

|                     |   frauds |   LightGBM |   Rules engine |   LightGBM dollar recall |
|:--------------------|---------:|-----------:|---------------:|-------------------------:|
| card_testing        |      104 |      1     |          0.048 |                    1     |
| account_takeover    |       37 |      0.946 |          0.081 |                    0.959 |
| point_of_compromise |       50 |      0.98  |          0.16  |                    0.991 |

## Example alerts

- $371.23 retail (point_of_compromise), score 0.99: Amount vs card's usual; New devices, last 24h; New merchant for card
- $311.14 retail (point_of_compromise), score 0.98: Amount vs card's usual; New device for card; New merchant for card
- $241.85 retail (card_testing), score 0.98: Amount vs card's usual; New devices, last 24h; New merchant for card
