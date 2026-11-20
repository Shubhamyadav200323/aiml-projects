# Individual Reflection: AIML-03 Large-Scale Personalised Recommendation Engine

**Shubham Yadav**

## What I owned

- Setting up and running the project on Windows: training, the test suite, the A/B simulation and the latency benchmark.
- Fixing a test failure that only happened on Windows (below).
- Exploring the API and the demo UI: personalised feeds, cold-start users, similar items and the A/B report.
- The screenshots, the demo recording, the presentation, and publishing everything to GitHub.

## What I learned

- **Why recommenders work in two stages.** Scoring every item for every user is impossible at scale (about 10 billion pairs in MovieLens 25M), so a fast two-tower model with a FAISS index finds a few hundred candidates and a deeper ranker orders only those.
- **Splitting by time.** A random split lets the model learn a user's future to predict their past. A time-based split gives honest numbers and also creates real cold-start users to test.
- **Ranking metrics.** I learned how Recall@K, NDCG@K, hit rate and catalogue coverage differ, and that a ranker can improve precision while showing less of the catalogue.
- **Cold start.** New users need a different strategy: content-based matching on genres blended with trending items worked better than popularity alone.
- **A/B testing properly.** The simulated test showed the two rankers were not significantly different (p = 0.39). The correct decision is not to ship on noise, and a power analysis tells you how much traffic you would need.

## Hardest parts and how I got past them

- **TensorFlow setup.** TensorFlow Recommenders needs Keras 2, so the environment variable `TF_USE_LEGACY_KERAS=1` has to be set before running anything.
- **A test that failed only on Windows.** The cache test stored an entry that expires immediately, waited 10 ms, and expected it to be gone. On Windows the clock only ticks about every 15 ms, so the entry still looked valid. The root cause was that the cache treated "expires exactly now" as still valid; changing `<` to `<=` fixed the logic, and a longer wait in the test made it robust. It taught me that tests can expose platform-specific timing assumptions.
- **The Spark test.** It is skipped on my laptop because Spark needs Java; it runs in the CI pipeline instead.
- **Keeping the Git history clean.** I learned to fold small fixes and screenshots into existing commits with `git commit --fixup` and `git rebase --autosquash`, and to sign commits with an SSH key so GitHub marks them Verified.

## What I would do next

Run the full pipeline on MovieLens 25M with the Spark job, add diversity-aware re-ranking to recover catalogue coverage, and learn how to serve the models through a real Triton server instead of in-process.
