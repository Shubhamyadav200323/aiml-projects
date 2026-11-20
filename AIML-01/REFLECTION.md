# Individual Reflection: AIML-01 Real-Time Fraud Detection Platform

**Shubham Yadav**

## What I owned

- Setting up and running the project on Windows: environment, data analysis, training, tests and the latency benchmark.
- Running the API and the analyst dashboard, streaming transactions through them, and reviewing fraud cases.
- Debugging the problems I hit along the way (below).
- The screenshots, the demo recording, the presentation, and publishing the repository with signed commits.

## What I learned

- **Accuracy is the wrong metric for fraud.** With only 0.52% fraud, a model that never flags anything is 99.5% accurate. I learned to read PR-AUC and recall at a fixed precision instead, and why the blended model was chosen even though its PR-AUC is slightly lower than XGBoost alone: it catches more fraud at the same 90% precision.
- **Training/serving skew and time leakage.** Features must be computed the same way for training and for live scoring, and only from past events. The project has tests for both, and seeing them pass made the idea concrete.
- **Why a graph helps.** Fraud rings share devices and cards, so a single transaction can look normal while the account's neighbourhood does not. The graph model alone is weak, but blended with XGBoost it adds value.
- **Three-tier architecture in practice.** The dashboard only talks to the API, the API and the Kafka consumer share one scoring class, and all state lives in the data tier. That separation is what makes each part testable and scalable.
- **Engineering hygiene.** Virtual environments per project, environment variables like `PYTHONPATH`, automated tests, CI, and signed Git commits.

## Hardest parts and how I got past them

- **Python versions on Windows.** My default Python was 3.13; the ML libraries needed 3.11, so I created the virtual environment explicitly with `py -3.11`.
- **Ports and services.** I first opened the dashboard's port (8501) looking for the API docs, and later got 404s on the API's root URL. Understanding which service runs where (API on 8000 at `/docs`, dashboard on 8501) cleared it up.
- **An empty-looking dashboard.** The p99 latency showed 0.0 ms because the API had restarted and its in-memory metrics reset, while the cases were stored in the database. I learned the difference between in-memory and persisted state.
- **Recording.** My first screen recording had no sound because the microphone was off in the recorder; I now test a short clip before recording.

## What I would do next

Run the pipeline on the real PaySim and IEEE-CIS datasets, deploy the stack on Minikube end to end, and study how the GraphSAGE model aggregates neighbour information in more depth so I could modify it myself.
