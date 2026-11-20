# Exploratory Data Analysis

* Transactions: **108,815**
* Fraudulent: **563** (0.517%) -> imbalance ratio 1:192
* Users: 3,000 | Devices: 6,025 | Cards: 5,045
* Time span: 2026-01-01 00:00:30.119988530 -> 2026-01-31 00:00:00

## Why accuracy is the wrong metric

A model that predicts *legit* for everything scores **99.48% accuracy** while catching zero fraud. We therefore optimise and report **PR-AUC** (average precision), whose no-skill baseline equals the fraud rate (**0.0052**).

## Amount by class

| class | mean | 50% | max |
|---|---|---|---|
| legit | 60.44 | 35.85 | 8,448 |
| fraud | 236.1 | 144.1 | 1,849 |

## Fraud rate by channel

| channel | legit_share | fraud_rate |
|---|---|---|
| PAYMENT | 0.9873 | 0.01267 |
| CASH_OUT | 0.9879 | 0.0121 |
| TRANSFER | 0.9916 | 0.008364 |
| ONLINE | 0.9961 | 0.003917 |
| CARD_PRESENT | 0.9993 | 0.000663 |

## Shared devices (graph signal)

* Devices used by >1 account: 167
* Fraud rate on shared devices: **6.01%** vs 0.52% overall

Shared infrastructure is the motivation for the graph neural network stage.
