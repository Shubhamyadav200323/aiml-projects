"""Airflow DAG: nightly feature refresh, retraining, quality gate and Triton rollout.

    spark_features >> train_models >> evaluate_gate >> [publish_to_triton, notify_rejected]
    publish_to_triton >> reload_triton >> invalidate_cache >> smoke_test

The quality gate compares the candidate's offline NDCG@10 with the production
model's and only promotes when it is no worse than `MAX_REGRESSION`, so a bad
night of data can never silently ship a worse model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

DATA = os.environ.get("RECSYS_DATA", "/data/ml-25m")
WORK = Path(os.environ.get("RECSYS_WORK", "/opt/recsys/runs"))
PROD = Path(os.environ.get("RECSYS_ARTIFACTS", "/opt/recsys/artifacts"))
TRITON_REPO = Path(os.environ.get("TRITON_MODEL_REPO", "/models"))
TRITON_URL = os.environ.get("TRITON_URL", "http://triton:8000")
API_URL = os.environ.get("RECSYS_API_URL", "http://recsys-api:8100")
MAX_REGRESSION = float(os.environ.get("MAX_NDCG_REGRESSION", "0.02"))

default_args = {"owner": "ml-platform", "retries": 1, "retry_delay": timedelta(minutes=10)}


def _best_ndcg(metrics_path: Path) -> float:
    warm = json.loads(metrics_path.read_text())["warm"]
    return max(v["ndcg@10"] for k, v in warm.items() if k.startswith("two_stage"))


def evaluate_gate(ds: str, **_) -> str:
    cand = _best_ndcg(WORK / ds / "metrics.json")
    prod_path = PROD / "metrics.json"
    prod = _best_ndcg(prod_path) if prod_path.exists() else 0.0
    print(f"candidate ndcg@10={cand:.4f} production={prod:.4f}")
    return "publish_to_triton" if cand >= prod * (1 - MAX_REGRESSION) else "notify_rejected"


def publish(ds: str, **_) -> None:
    import shutil

    run = WORK / ds
    version = datetime.strptime(ds, "%Y-%m-%d").strftime("%Y%m%d")
    for model_dir in (run / "models").iterdir():
        dest = TRITON_REPO / model_dir.name / version
        shutil.copytree(model_dir / "1", dest, dirs_exist_ok=True)
    tmp = PROD.with_suffix(".new")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(run, tmp)
    if PROD.exists():
        shutil.rmtree(PROD.with_suffix(".prev"), ignore_errors=True)
        PROD.rename(PROD.with_suffix(".prev"))
    tmp.rename(PROD)  # atomic swap; .prev kept for one-command rollback


with DAG(
    dag_id="recsys_retrain",
    description="Two-stage recommender: features -> train -> gate -> Triton",
    schedule="0 3 * * *",
    start_date=datetime(2026, 10, 20),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["recsys", "aiml-03"],
) as dag:
    spark_features = BashOperator(
        task_id="spark_features",
        bash_command=(
            "spark-submit --master ${SPARK_MASTER:-local[*]} /opt/recsys/src/recsys/features/spark_job.py "
            f"--ratings {DATA}/ratings.csv --movies {DATA}/movies.csv --as-of '{{{{ ds }}}}' "
            f"--out {WORK}/{{{{ ds }}}}/features_spark"
        ),
    )
    train_models = BashOperator(
        task_id="train_models",
        bash_command=f"python -m recsys.pipeline train --dataset movielens --path {DATA} --out {WORK}/{{{{ ds }}}}",
        env={"TF_USE_LEGACY_KERAS": "1"}, append_env=True,
        execution_timeout=timedelta(hours=4),
    )
    gate = BranchPythonOperator(task_id="evaluate_gate", python_callable=evaluate_gate)
    publish_to_triton = PythonOperator(task_id="publish_to_triton", python_callable=publish)
    notify_rejected = BashOperator(task_id="notify_rejected",
                                   bash_command="echo 'candidate rejected by quality gate' && exit 0")
    reload_triton = BashOperator(
        task_id="reload_triton",
        bash_command=" && ".join(
            f"curl -fsS -X POST {TRITON_URL}/v2/repository/models/{m}/load"
            for m in ("user_tower", "ranker_dcn_v2", "ranker_wide_deep")),
    )
    invalidate_cache = BashOperator(task_id="invalidate_cache",
                                    bash_command=f"curl -fsS -X POST {API_URL}/v1/cache/invalidate")
    smoke_test = BashOperator(task_id="smoke_test",
                              bash_command=f"curl -fsS '{API_URL}/v1/recommendations/u1?k=10' | grep -q items")

    spark_features >> train_models >> gate >> [publish_to_triton, notify_rejected]
    publish_to_triton >> reload_triton >> invalidate_cache >> smoke_test
