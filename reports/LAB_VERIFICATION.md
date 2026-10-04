# Prayogshala lab verification (4 October 2026)

Project path: ~/workspace/heart-disease-mlops
Environment: Rocky Linux 9.5 VM; Python 3.12 environment at ~/workspace/.venv
Data: UCI processed Cleveland file, 303 rows

## Checks actually run in this VM

- python -m pytest -q: 8 passed, 1 warning (httpx/Starlette deprecation)
- ruff check .: All checks passed
- python -m heartlab.data and python -m heartlab.eda: cleaned CSV and three EDA plots written
- python -m heartlab.train: two candidate runs plus selected-model run logged to local MLflow SQLite store
- Selected logistic regression; test accuracy 0.8852459016, precision 0.8387096774, recall 0.9285714286, ROC-AUC 0.9664502165
- docker build -t heartlab:local .: succeeded
- Docker /health: {"status":"ready"}
- Docker /predict for sample_request.json: prediction 0, confidence 0.8345, positive probability 0.1655
- Minikube v1.35.0 with Docker driver: cluster started
- Kubernetes deployment after explicit runAsUser: 1000 fix: rollout successful, 2/2 replicas available, two pods Running and ready
- Kubernetes service via kubectl port-forward: /health ready; /predict returned the same sample result
- Kubernetes /metrics: Prometheus text metrics returned

Evidence: screenshots/kubernetes.png is a direct screenshot taken in the lab VM. Training details are in training.log and artifacts/metadata.json. This is a local Minikube demonstration; the LoadBalancer external IP remained pending, so service access used port-forward. GitHub Actions and the lab walkthrough video are verified below.

## Public CI verification

Commit `b0f1f75` was pushed from this Prayogshala VM to the public coursework repository. The first GitHub Actions run completed successfully in 2 minutes 9 seconds. A screenshot taken in this VM is saved at `screenshots/github-actions.png`. A short silent VM screen recording is saved at demo/pipeline_walkthrough.webm`.
