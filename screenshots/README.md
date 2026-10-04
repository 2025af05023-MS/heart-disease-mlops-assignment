# Evidence to capture in the virtual lab

Save genuine screenshots here, with the exact commands and dates recorded in the
report. Suggested names:

1. `01-eda.png` - EDA figures opened in the lab.
2. `02-mlflow.png` - both model runs and their logged metrics.
3. `03-tests.png` - `ruff check` and `pytest -q` passing.
4. `04-ci.png` - successful GitHub Actions run after a real repository push.
5. `05-docker.png` - `docker build` completion and local prediction response.
6. `06-kubernetes.png` - rollout status, pods, service, and endpoint response.
7. `07-monitoring.png` - `/metrics` after several prediction calls.

Do not use generated or example screenshots as evidence. If a step cannot be run in
the current lab, state the reason and omit its screenshot.
