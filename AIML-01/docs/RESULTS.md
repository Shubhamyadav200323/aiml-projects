# Results

Reproduce with `make train && make loadtest && make test`. These numbers come from the
synthetic PaySim-style dataset (108,815 transactions, 0.52% fraud, 12 planted rings and
60 account takeovers), with a temporal 70/15/15 split. Re-run on PaySim or IEEE-CIS with
`--dataset paysim --path ...` and expect lower absolute numbers on real data.

## Model quality (held-out test period, 16,323 transactions)

| Model | PR-AUC | ROC-AUC | Recall @ precision 90% | Precision @ top 1% | F1 |
|---|---|---|---|---|---|
| No-skill baseline | 0.011 | 0.500 | 0 | 0.011 | - |
| XGBoost (class-weighted) | **0.966** | 0.997 | 0.918 | 0.976 | 0.931 |
| GraphSAGE (account score only) | 0.333 | 0.979 | 0.000 | 0.368 | 0.404 |
| **Blend (w_xgb = 0.5)** | 0.959 | **0.998** | **0.945** | **0.988** | **0.940** |

Takeaways

* The graph model alone is a weak transaction classifier (it only knows *who*, not
  *what*), but blending it in raises recall at 90% precision by **+2.7 points** and
  precision in the analyst top-1% queue by **+1.2 points**. That matters because the
  blend catches ring members whose individual transactions look normal.
* 12 fraud rings were recovered by connected components over high-risk accounts.

## Serving performance (single core, in-process, 2,000 events)

| p50 | p95 | p99 | Throughput |
|---|---|---|---|
| 0.94 ms | 1.10 ms | 1.38 ms | ~1,035 TPS per core |

## Test suite

`pytest`: **23 tests passed** covering schema/imbalance, online/offline feature parity,
temporal leakage, velocity window semantics, XGBoost and GNN quality, artifact
round-trip, the API contract, validation, the case feedback loop, Kafka replay and
consume with DLQ, PSI drift, and the retrain/promotion policy.
