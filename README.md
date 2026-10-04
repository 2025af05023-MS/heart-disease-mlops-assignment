# Heart disease prediction: an MLOps experiment

This is an educational binary classifier using the **processed Cleveland** portion of
the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).
The original 14th column, `num`, ranges from 0 to 4. I map 0 to *absence* and
1–4 to *presence*. The 13 inputs are the UCI fields, not a diagnosis or a calibrated
clinical risk score. The `/predict` probability describes this model's estimate for
the historical dataset; it is **not suitable for medical decisions**.

## What is in the project

| Requirement | Implementation |
|---|---|
| Acquisition and cleaning | `src/heartlab/data.py`, public UCI URL, `?` to missing, binary target |
| EDA | `src/heartlab/eda.py`, distributions, class balance, correlation heatmap |
| Modelling | `src/heartlab/train.py`, logistic regression and random forest, 5-fold stratified CV |
| Tracking | MLflow SQLite backend with parameters, metrics, figures, model artifact |
| Serving | FastAPI with `/predict`, `/health`, `/metrics` |
| CI | `.github/workflows/ci.yml`: lint, tests, train, Docker build, smoke test, artifacts |
| Deployment | `k8s/deployment.yaml`: two replicas and LoadBalancer service |
| Monitoring | Structured request logs without feature values and Prometheus counters/latency |

## Run in Prayogshala virtual lab

The lab manual says the persistent folder is `~/workspace`. Keep the browser tab open.
The session can idle out after 15 minutes, and a stopped VM can terminate after
60 minutes. **Copy or clone this entire project under `~/workspace` before working.**
Installing packages is temporary; the files in `~/workspace` persist. The lab manual
describes launching the course lab from the Prayogshala portal through BITS SSO.

```bash
cd ~/workspace/heart-disease-mlops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m heartlab.data
python -m heartlab.eda
ruff check src tests
pytest -q
python -m heartlab.train | tee training.log
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Visit `http://127.0.0.1:5000` in the lab browser to inspect MLflow runs. The
download step creates both the raw UCI file and `data/cleveland_clean.csv`.
The training step creates `artifacts/model.joblib`, `artifacts/metadata.json`, EDA
figures, confusion matrix, ROC curve, a local `mlflow.db` tracking database,
and run artifacts under `mlruns/`. The local verification database is not
shipped because its artifact URIs are machine-specific; `reports/MLFLOW_RUNS.md`
records its three completed runs. Run training in the lab to create lab-local
tracking paths. The full preprocessing pipeline is saved
inside the joblib model. All imputation and scaling are fitted inside CV folds.

Run the API in a second terminal:

```bash
cd ~/workspace/heart-disease-mlops
source .venv/bin/activate
export PYTHONPATH=src
uvicorn heartlab.api:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' -d @sample_request.json
curl -s http://127.0.0.1:8000/metrics | head
```

The sample row is a publicly published UCI example. Never send actual personal
health records to this classroom service. API logs include status and latency only.

## Container and local Kubernetes

After the training step, and only if Docker is available in the lab:

```bash
docker build -t heartlab-api:latest .
docker run --rm -p 8000:8000 heartlab-api:latest
```

Use the same `curl` prediction in another terminal. For Minikube, start a cluster
and load the image into it. These commands require Docker and Minikube installed
in the temporary VM session:

```bash
minikube start --driver=docker
minikube image load heartlab-api:latest
minikube kubectl -- apply -f k8s/deployment.yaml
minikube kubectl -- rollout status deployment/heartlab-api
minikube kubectl -- get pods,service
minikube tunnel
```

`minikube tunnel` runs in a separate terminal and needs to stay active. The lab
has Minikube even when `kubectl` is absent, so `minikube kubectl --` is used above.
If the LoadBalancer external IP is unavailable in the virtual lab, verify the deployment
using `minikube kubectl -- port-forward service/heartlab-api 8000:8000` and explain that
environment limitation in the report. Do not describe it as a public cloud URL.

## Experimental design

I fix seed 42, stratify an 80/20 split, and reserve the test set until the two
model families have been compared using five stratified CV folds on training data.
Each model has its own preprocessing, imputation, and grid search. I select by mean
CV ROC-AUC, then evaluate the selected model once on the test set. Accuracy,
precision, recall, and ROC-AUC are logged. The 0.5 decision threshold is fixed
before looking at test results. The small historical sample, site/time drift,
possible selection bias, and missing `ca`/`thal` values limit generalization.

### Observed local run (4 October 2026)

The reproducible local run used 242 training records and 61 held-out records.
Five-fold mean ROC-AUC was **0.9025** for logistic regression and **0.8988** for
random forest. Logistic regression was selected by the predeclared rule. Its
held-out accuracy was **0.8852**, precision **0.8387**, recall **0.9286**, and
ROC-AUC **0.9665**. These are small-sample classroom results, not estimates of
clinical safety. `artifacts/metadata.json` contains the full precision, seed,
parameters, and MLflow run IDs. Locally, Ruff passed, all eight Pytest tests
passed, and an actual saved-model API request returned HTTP 200.

## Evidence for submission

The `screenshots/README.md` file lists the real screenshots to capture in the
lab. The final report must use numbers from `artifacts/metadata.json`, a link to
the student's actual GitHub repository, and screenshots from actual CI, Docker,
MLflow, and Kubernetes runs. A demo outline is in `demo/recording_script.md`.
The completed VM deployment is documented in reports/LAB_VERIFICATION.md and screenshots/kubernetes.png; GitHub Actions passed on the first public push. This VM contains screenshots/github-actions.png and a short silent demo/pipeline_walkthrough.webm.

The coursework repository is `https://github.com/2025af05023-MS/heart-disease-mlops-assignment`.
To publish updates after reviewing them, use these commands from this folder.

```bash
git init -b main
git add .
git commit -m "Add Cleveland heart disease MLOps experiment"
git remote add origin https://github.com/2025af05023-MS/heart-disease-mlops-assignment.git
git push -u origin main
```

## Source and license

UCI Machine Learning Repository, Heart Disease, DOI
[`10.24432/C52P4X`](https://doi.org/10.24432/C52P4X). The UCI page lists the
dataset's attribution and use terms. This project code was written for this
assignment; dataset provenance is recorded in the report and metadata.
