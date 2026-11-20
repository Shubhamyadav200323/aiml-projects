"""Spark feature job for full-scale data (MovieLens 25M: 25M ratings, 162k users).

    spark-submit src/recsys/features/spark_job.py --ratings data/ml-25m/ratings.csv \
        --movies data/ml-25m/movies.csv --as-of 2018-01-01 --out data/features

Writes `interactions/`, `user_features/` and `item_features/` as parquet,
mirroring `recsys.features.tables.build_feature_tables` column for column.
"""

from __future__ import annotations

import argparse

from recsys.data.movielens import GENRES
from recsys.features.tables import HISTORY_LEN, POP_HALF_LIFE_DAYS


def build(spark, ratings, movies, as_of: str, min_rating: float = 4.0):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    # Epoch seconds are instants but `as_of` is a wall-clock string: pin the session to UTC,
    # otherwise the cutoff silently shifts by the cluster's local offset.
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    as_of_ts = F.to_timestamp(F.lit(as_of))
    r = (ratings
         .withColumn("timestamp", F.col("timestamp").cast("timestamp"))
         .where(F.col("timestamp") < as_of_ts)
         .withColumn("label", (F.col("rating") >= min_rating).cast("int")))
    m = movies.withColumn("genres", F.split("genres", r"\|"))
    for g in GENRES:
        m = m.withColumn(f"g_{g}", F.array_contains("genres", g).cast("double"))

    # ---- item features -------------------------------------------------
    age_days = (F.unix_timestamp(as_of_ts) - F.unix_timestamp("timestamp")) / 86_400.0
    item_stats = (r.withColumn("_decay", F.pow(F.lit(0.5), age_days / POP_HALF_LIFE_DAYS))
                  .groupBy("item_id")
                  .agg(F.log1p(F.count("*")).alias("pop_log"),
                       F.log1p(F.sum("_decay")).alias("pop_recent"),
                       F.avg("rating").alias("item_avg_rating")))
    global_avg = r.agg(F.avg("rating")).first()[0]
    item_features = (m.select("item_id", *[f"g_{g}" for g in GENRES])
                     .join(item_stats, "item_id", "left")
                     .fillna({"pop_log": 0.0, "pop_recent": 0.0, "item_avg_rating": global_avg}))

    # ---- user features -------------------------------------------------
    pos = r.where("label = 1").join(m.select("item_id", *[f"g_{g}" for g in GENRES]), "item_id")
    aff = pos.groupBy("user_id").agg(*[F.sum(f"g_{g}").alias(f"aff_{g}") for g in GENRES])
    total = sum(F.col(f"aff_{g}") for g in GENRES)
    aff = aff.select("user_id", *[(F.col(f"aff_{g}") / F.when(total == 0, 1).otherwise(total)).alias(f"aff_{g}")
                                  for g in GENRES])
    w = Window.partitionBy("user_id").orderBy(F.col("timestamp").desc())
    hist = (pos.withColumn("rk", F.row_number().over(w)).where(F.col("rk") <= HISTORY_LEN)
            .groupBy("user_id")
            .agg(F.sort_array(F.collect_list(F.struct("timestamp", "item_id"))).alias("h"))
            .select("user_id", F.col("h.item_id").alias("history")))
    base = r.groupBy("user_id").agg(F.count("*").cast("double").alias("n_interactions"),
                                    F.sum("label").cast("double").alias("n_positives"),
                                    F.avg("rating").alias("avg_rating"))
    user_features = (base.join(aff, "user_id", "left").fillna(0.0)
                     .join(hist, "user_id", "left")
                     .withColumn("history", F.coalesce("history", F.array().cast("array<string>"))))
    return r, user_features, item_features


def main() -> None:  # pragma: no cover - cluster entrypoint
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--ratings", required=True)
    p.add_argument("--movies", required=True)
    p.add_argument("--as-of", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    spark = SparkSession.builder.appName("recsys-features").getOrCreate()
    ratings = (spark.read.csv(a.ratings, header=True, inferSchema=True)
               .select(F.concat(F.lit("u"), "userId").alias("user_id"),
                       F.concat(F.lit("i"), "movieId").alias("item_id"),
                       F.col("rating").cast("double"), F.col("timestamp").cast("long")))
    movies = (spark.read.csv(a.movies, header=True, inferSchema=True)
              .select(F.concat(F.lit("i"), "movieId").alias("item_id"), "title", "genres"))
    inter, users, items = build(spark, ratings, movies, a.as_of)
    inter.write.mode("overwrite").partitionBy("label").parquet(f"{a.out}/interactions")
    users.write.mode("overwrite").parquet(f"{a.out}/user_features")
    items.write.mode("overwrite").parquet(f"{a.out}/item_features")
    spark.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
