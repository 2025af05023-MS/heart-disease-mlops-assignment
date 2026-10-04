# Short demo recording outline (about 3 minutes)

1. Show the project under `~/workspace` and briefly explain the UCI source.
2. Run `pytest -q`, then show the MLflow comparison and selected model.
3. Show the saved figures and the held-out metrics in `artifacts/metadata.json`.
4. Show the Docker image and make one `POST /predict` request.
5. Show the Kubernetes rollout and LoadBalancer or port-forward endpoint.
6. Call `/metrics` and point to request count and latency; mention that raw
   patient fields are excluded from logs.
7. State the limits: 303 historical rows, educational use, no clinical validation.

Record the actual screen in the lab or on your computer after the working system
is visible. Save the `.mp4` in `demo/` and include it in the final repository.
