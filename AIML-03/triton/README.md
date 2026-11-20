# Triton model repository

`python -m recsys.pipeline train` exports SavedModels in Triton's layout
(`artifacts/models/<model>/<version>/model.savedmodel`). Copy or mount them next to the
`config.pbtxt` files here (the Airflow DAG does this with a date-stamped version):

```bash
cp -r ../artifacts/models/* model_repository/          # keeps config.pbtxt
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models --model-control-mode=explicit \
  --load-model=user_tower --load-model=ranker_dcn_v2 --load-model=ranker_wide_deep
```

Point the API at it with `TRITON_URL=localhost:8000`; `recsys.serving.models.TritonModels`
sends the same tensors as the in-process backend. Dynamic batching on the rankers groups
concurrent requests' 300-candidate blocks into larger batches.
