from pathlib import Path
import json, math, os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

if Path(__file__).resolve().parent.name == 'reports':
    BASE=Path(__file__).resolve().parents[1]
    OUT=BASE/'reports/MLOps_Experimental_Learning_Report.pdf'
else:
    BASE=Path('outputs/heart-disease-mlops')
    OUT=Path('outputs/MLOps_Experimental_Learning_Report.pdf')
student=os.getenv('STUDENT_NAME','MAHENDRA SINGH / 2025AF05023')
repository=os.getenv('GITHUB_REPO_URL','https://github.com/2025af05023-MS/heart-disease-mlops-assignment')
data=pd.read_csv(BASE/'data/cleveland_clean.csv')
metadata_path=BASE/'artifacts/metadata.json'
meta=json.loads(metadata_path.read_text()) if metadata_path.exists() else None

FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if Path(FONT).exists():
    pdfmetrics.registerFont(TTFont('Arial',FONT))
    pdfmetrics.registerFont(TTFont('Arial-Bold',BOLD))
    reg,bold='Arial','Arial-Bold'
else:
    reg,bold='Helvetica','Helvetica-Bold'
navy=colors.HexColor('#15324d'); blue=colors.HexColor('#2b6ca3'); light=colors.HexColor('#eaf2f8'); gray=colors.HexColor('#5a6670')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName=bold,fontSize=22,leading=27,textColor=navy,spaceAfter=18))
styles.add(ParagraphStyle(name='H1X',fontName=bold,fontSize=16,leading=20,textColor=navy,spaceAfter=12))
styles.add(ParagraphStyle(name='H2X',fontName=bold,fontSize=11,leading=14,textColor=blue,spaceBefore=8,spaceAfter=5))
styles.add(ParagraphStyle(name='BodyX',fontName=reg,fontSize=10,leading=15,spaceAfter=10))
styles.add(ParagraphStyle(name='SmallX',fontName=reg,fontSize=8.8,leading=12,spaceAfter=6))
styles.add(ParagraphStyle(name='NoteX',fontName=reg,fontSize=9,leading=13,textColor=gray,spaceAfter=7))
styles.add(ParagraphStyle(name='CoverX',fontName=reg,fontSize=11,leading=16,textColor=gray,spaceAfter=13))
P=lambda s,sty='BodyX': Paragraph(s,styles[sty])
S=lambda h=6: Spacer(1,h)

def heading(title,kicker=None):
    out=[]
    if kicker: out.append(P(kicker.upper(),'SmallX'))
    out.append(P(title,'H1X'))
    return out

def bullets(items):
    return [P('• '+x) for x in items]

def table(rows,widths=None,header=True):
    t=Table([[P(str(x),'SmallX') for x in row] for row in rows],colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
    ts=[('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.8,blue),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]
    if header: ts += [('BACKGROUND',(0,0),(-1,0),light)]
    t.setStyle(TableStyle(ts)); return t

def page(canvas,doc):
    canvas.saveState(); w,h=A4
    canvas.setStrokeColor(colors.HexColor('#ccd7df')); canvas.line(18*mm,h-18*mm,w-18*mm,h-18*mm)
    canvas.setFont(reg,7.5); canvas.setFillColor(gray)
    canvas.drawString(18*mm,12*mm,'AIMLCZG523 • MLOps Assignment 01 • Cleveland Heart Disease')
    canvas.drawRightString(w-18*mm,12*mm,f'Page {doc.page}')
    canvas.restoreState()

story=[]
# 1 cover
story += [S(25),P('MLOps Experimental Learning<br/>Assignment 01','TitleX'),P('End-to-end model development, CI/CD and deployment using the UCI Cleveland Heart Disease data','CoverX')]
story += [table([['Course','AIMLCZG523 — Machine Learning Operations'],['Submission','Student name / ID: '+student],['Environment','BITS Prayogshala virtual lab; persistent path ~/workspace'],['Repository',repository],['Date','4 October 2026']], [36*mm,135*mm],False),S(20)]
story += [P('Abstract','H2X'),P('I built a small, reproducible classification workflow around the public Cleveland heart disease records. The aim is to demonstrate the MLOps lifecycle rather than to present a clinical tool: acquisition and EDA, two comparable modelling pipelines, MLflow experiment tracking, a FastAPI service, GitHub Actions checks, Docker packaging, Kubernetes manifests and basic operational metrics.')]
story += [P('Status note','H2X'),P('I repeated preparation and training in the Prayogshala VM. Eight tests and Ruff passed there. The Docker image served a real prediction; Minikube ran two ready API pods, and the service returned a prediction and Prometheus metrics. The GitHub Actions workflow passed on the first public push; its screenshot is included on page 8. A short silent lab screen recording is supplied at demo/pipeline_walkthrough.webm.')]
story += [P('Key design choice','H2X'),P('All imputation, one-hot encoding and scaling live inside each scikit-learn pipeline. The 20% hold-out set is used only after five-fold cross-validation selects the model family. This avoids using test data during feature preparation or tuning.')]
story += [PageBreak()]
# 2 data
story += heading('Problem statement and dataset','01 / Context')
story += [P('The task is binary classification: given 13 routinely recorded fields from the processed Cleveland data, predict whether the original UCI outcome indicates disease presence. The historical outcome <b>num = 0</b> is mapped to 0, while values 1–4 map to 1. This follows the dataset documentation. The API returns a model score and class label, not a medical diagnosis.')]
story += [P('The assignment brief refers to “14+ features,” while the official processed Cleveland release has 14 columns total: 13 predictors plus the outcome. I use the source schema as published rather than inventing an extra predictor.','NoteX')]
story += [table([['Source','UCI Machine Learning Repository, Heart Disease (processed Cleveland)'],['Records','303 rows; 13 features and one outcome'],['Target balance','164 absence (54.1%); 139 presence (45.9%)'],['Missing inputs','4 missing ca values; 2 missing thal values'],['Attribution','Janosi et al. (1989), DOI 10.24432/C52P4X; CC BY 4.0']], [36*mm,135*mm])]
story += [S(10),P('Acquisition and cleaning','H2X'),P('The source URL and a download script are in <font face="Courier">src/heartlab/data.py</font>. The original file is saved verbatim. The cleaner checks the 14-field schema and target range, turns UCI question marks into missing values, converts columns to numeric values and writes a CSV with a binary target. It does not fill missing values before the train/test split. A copy of the cleaned CSV is included in the project.')]
story += [P('Practical data caution','H2X'),P('The dataset has no patient identifiers in the processed file, but the features still describe health. For this classroom system, examples should come from the published dataset and API logs should omit raw input values. The sample is small and historical, so modern clinical or population performance cannot be inferred from these scores.')]
story += [PageBreak()]
# 3 EDA
story += heading('Exploratory data analysis','02 / Dataset review')
story += [P('The EDA script produces three figures: selected feature histograms by target, a class-balance chart, and a Pearson correlation heatmap. I use the heatmap as a screening view only because integers that code categories do not have a meaningful numeric distance. The code saves high-resolution PNG files for the report and MLflow.')]
# bar chart
fig=Drawing(430,145)
fig.add(String(0,130,'Target balance (303 records)',fontName=bold,fontSize=10,fillColor=navy))
for x,n,c,label in [(65,164,blue,'Absent'),(250,139,colors.HexColor('#db8b42'),'Present')]:
    height=n/180*90
    fig.add(Rect(x,25,95,height,fillColor=c,strokeColor=None))
    fig.add(String(x+47,14,label,textAnchor='middle',fontName=reg,fontSize=9))
    fig.add(String(x+47,30+height,str(n),textAnchor='middle',fontName=bold,fontSize=9))
story += [fig,S(10)]
heat=Drawing(430,198)
heat.add(String(0,183,'Selected Pearson correlations',fontName=bold,fontSize=10,fillColor=navy))
fields=['age','trestbps','chol','thalach','oldpeak','target']
short={'age':'age','trestbps':'BP','chol':'chol','thalach':'HR','oldpeak':'ST','target':'target'}
matrix=data[fields].corr()
cell=25; startx=83; starty=23
for i,name in enumerate(fields):
    heat.add(String(startx+i*cell+cell/2,8,short[name],textAnchor='middle',fontName=reg,fontSize=7))
    heat.add(String(0,starty+(5-i)*cell+9,name,fontName=reg,fontSize=8))
    for j,col in enumerate(fields):
        value=float(matrix.loc[name,col])
        if value>=0:
            c=colors.Color(1-0.10*value,1-0.45*value,1-0.70*value)
        else:
            v=-value; c=colors.Color(1-0.72*v,1-0.48*v,1-0.12*v)
        heat.add(Rect(startx+j*cell,starty+(5-i)*cell,cell-1,cell-1,fillColor=c,strokeColor=None))
heat.add(String(255,100,'Warm: positive',fontName=reg,fontSize=8,fillColor=gray))
heat.add(String(255,86,'Cool: negative',fontName=reg,fontSize=8,fillColor=gray))
story += [heat,S(7)]
story += [P('Checks from the cleaned file','H2X'),table([['Check','Result / interpretation'],['Rows and fields','303 rows; no dropped records; 13 input features'],['Missing values','ca: 4; thal: 2; median/mode imputation deferred to pipeline'],['Outcome','164 negative, 139 positive; stratification preserves this mix'],['Visual review','Age, cholesterol and maximum heart rate histograms are produced by eda.py']], [38*mm,133*mm])]
story += [S(9),P('The class balance is reasonably close but not perfectly equal. I therefore retain stratified folds and report recall alongside accuracy. The visual EDA is descriptive and does not prove causal effects or justify medical use.')]
story += [PageBreak()]
# 4 features
story += heading('Preprocessing and feature engineering','03 / Reproducibility')
story += [P('I divided the 13 inputs into five numerical variables (age, resting blood pressure, cholesterol, maximum heart rate and oldpeak) and eight coded categorical variables. The numerical branch fits a median imputer and standard scaler; the categorical branch fits a most-frequent imputer and one-hot encoder. Unknown categories at inference are ignored so the service can fail gracefully rather than crashing.')]
story += [table([['Stage','Implementation','Reason'],['Split','80/20 stratified, random_state=42','Keeps a single untouched test set'],['Numerical','Median imputer + StandardScaler','Handles missing values and LR scale sensitivity'],['Categorical','Mode imputer + OneHotEncoder','Avoids false numerical ordering'],['CV','StratifiedKFold(5), shuffled, seed 42','Comparable and repeatable model selection'],['Artifact','joblib dump of complete Pipeline','Same transformations in training and serving']], [28*mm,77*mm,66*mm])]
story += [S(14),P('Why the pipeline matters','H2X'),P('A tempting shortcut is to calculate medians, scales or encodings once on the whole CSV before cross-validation. That would leak information from validation folds. Here scikit-learn refits each transformer within every fold. The serialized final pipeline contains its fitted transformations, so a fresh JSON request follows the same steps as training.')]
story += [P('Schema contract','H2X'),P('The API expects the 13 named fields and rejects unknown fields and out-of-range values with a validation error. ca and thal can be null because the original data has missing values. The training CSV and request model are deliberately small and easy to inspect.')]
story += [PageBreak()]
# 5 models
story += heading('Models, tuning and evaluation','04 / Experiment design')
story += [P('The candidate families are logistic regression and random forest. The logistic regression grid varies inverse regularization strength C over 0.1, 1.0 and 10.0. The forest grid varies tree count (100/200), maximum depth (3/unrestricted), and minimum leaf size (1/3). Every combination is evaluated with the same five stratified folds.')]
story += [table([['Metric','Why it is included'],['Accuracy','Overall correctness; can hide class-specific errors'],['Precision','How often a predicted positive is truly positive'],['Recall','How many positive cases are found'],['ROC-AUC','Ranking discrimination across thresholds; selection metric']], [40*mm,131*mm])]
story += [S(12),P('Selection rule','H2X'),P('The model with the higher <b>mean cross-validation ROC-AUC on the training split</b> is selected. Only then is its predicted probability evaluated on the held-out test split. The classification threshold is fixed at 0.5, and a confusion matrix plus ROC curve are saved. This separation keeps the test estimate from steering the selection.')]
if meta:
    rows=[['Candidate','ROC-AUC','Accuracy','Precision','Recall']]
    for name,v in meta['candidate_results'].items(): rows.append([name,f"{v['cv']['roc_auc']:.3f}",f"{v['cv']['accuracy']:.3f}",f"{v['cv']['precision']:.3f}",f"{v['cv']['recall']:.3f}"])
    story += [P('Observed five-fold CV results','H2X'),table(rows,[51*mm,30*mm,30*mm,30*mm,30*mm]),P('Selected: '+meta['selected_model']+'. Held-out metrics: '+', '.join(f'{k} {v:.3f}' for k,v in meta['test_metrics'].items())+'.')]
    cm=Image(str(BASE/'artifacts/confusion_matrix.png'),width=70*mm,height=56*mm)
    roc=Image(str(BASE/'artifacts/roc_curve.png'),width=70*mm,height=56*mm)
    figures=Table([[cm,roc]],colWidths=[85*mm,85*mm],hAlign='LEFT')
    story += [figures,P('Left: held-out confusion matrix. Right: held-out ROC curve. Both were generated by the completed training run.','NoteX')]
else:
    story += [P('Observed results','H2X'),P('Pending the actual lab training run. The completed values will be read from artifacts/metadata.json and the MLflow run history; no score is estimated here.','NoteX')]
story += [PageBreak()]
# 6 tracking
story += heading('Experiment tracking and packaging','05 / MLflow')
story += [P('MLflow is configured with a local SQLite tracking database, <font face="Courier">mlflow.db</font>. Each model family has a run with its best grid parameters, seed, training size, mean CV accuracy, precision, recall and ROC-AUC, plus the EDA plots. A final selection run records the chosen family, held-out metrics, serialized pipeline and diagnostic figures.')]
story += [table([['Artifact','Location / purpose'],['Raw source','data/processed.cleveland.data — exact UCI download'],['Clean CSV','data/cleveland_clean.csv — repeatable modelling input'],['Fitted model','artifacts/model.joblib — full preprocessing + classifier'],['Metadata','artifacts/metadata.json — split, seed, CV scores, test scores'],['Visuals','artifacts/eda, confusion_matrix.png, roc_curve.png'],['Run history','Generated by training in each environment; local summary in reports/MLFLOW_RUNS.md']], [47*mm,124*mm])]
story += [S(12),P('Clean-room repeat','H2X'),P('Install dependencies from requirements.txt, run the data module, run lint and tests, then run training. A fixed seed and explicit search grid keep the experiment repeatable. Exact bit-for-bit metrics may still depend on dependency versions and platform; the requirements file uses bounded versions rather than a full lockfile, which is a limitation worth addressing in a larger project.')]
story += [PageBreak()]
# 7 API
story += heading('API and container','06 / Serving')
story += [P('FastAPI loads the joblib pipeline on the first health or prediction request. POST /predict accepts the 13 feature names as JSON and returns the binary class, positive-class probability, a confidence value and model version. GET /health confirms that the model can be loaded; GET /metrics publishes Prometheus-format counters and a latency histogram.')]
story += [table([['Endpoint','Contract'],['POST /predict','200: prediction, confidence, probability_positive, model_version'],['GET /health','200 when model loads; 503 otherwise'],['GET /metrics','Prometheus text exposition'],['Validation','422 for malformed or out-of-range JSON'],['Logging','Event, status and duration only; no individual feature values']], [44*mm,127*mm])]
story += [S(12),P('Docker package','H2X'),P('The Dockerfile starts from python:3.11-slim, installs requirements, copies source and the trained model, then runs Uvicorn as a non-root user on port 8000. The model must be trained before docker build. A local smoke test uses the supplied sample_request.json. The CI workflow also builds and tests the image.')]
arch=Drawing(450,108)
arch.add(String(0,96,'System architecture',fontName=bold,fontSize=10,fillColor=navy))
boxes=[(0,'UCI CSV'),(91,'Cleaning'),(182,'CV + MLflow'),(273,'Model file'),(364,'FastAPI')]
for x,label in boxes:
    arch.add(Rect(x,43,82,34,fillColor=light,strokeColor=blue,strokeWidth=0.7))
    arch.add(String(x+41,56,label,textAnchor='middle',fontName=reg,fontSize=8.5,fillColor=navy))
for x in [82,173,264,355]:
    arch.add(Line(x,60,x+9,60,strokeColor=blue,strokeWidth=1.4))
    arch.add(Line(x+5,64,x+9,60,strokeColor=blue,strokeWidth=1.4))
    arch.add(Line(x+5,56,x+9,60,strokeColor=blue,strokeWidth=1.4))
arch.add(String(405,18,'Docker / Kubernetes',textAnchor='middle',fontName=reg,fontSize=8,fillColor=gray))
arch.add(String(405,5,'logs + /metrics',textAnchor='middle',fontName=reg,fontSize=8,fillColor=gray))
story += [S(10),arch]
story += [P('Example request','H2X'),P('<font face="Courier">curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json</font>','SmallX')]
story += [PageBreak()]
# 8 CI
story += heading('CI/CD workflow and tests','07 / Automation')
story += [P('The GitHub Actions workflow runs on push, pull request and manual dispatch. It installs dependencies, runs Ruff and Pytest, downloads the public dataset, trains the candidates with MLflow, builds the Docker image, sends a sample request to the container, and uploads the logs, model, figures and run store as workflow artifacts. Any failing lint, test, training, build or smoke-test command fails the job.')]
story += [table([['Gate','Evidence from workflow'],['Lint','ruff check src tests'],['Unit tests','Cleaning, invalid target, pipeline missing values, metrics, API response and validation'],['Training','Two CV searches plus saved model and metadata'],['Container','docker build followed by a real HTTP prediction'],['Artifacts','training.log, cleaned CSV, model/plots and mlruns']], [38*mm,133*mm])]
story += [S(12),P('Deployment handoff','H2X'),P('The manifest is applied separately in the lab after the tested image is available. This is deliberate because the course lab may not expose credentials for an automatic GitHub-to-cluster deployment. The workflow does not claim a deployment it has not performed. A production setup would add image publishing, an environment approval, secret management and automated rollout to a managed cluster.')]
story += [P('Successful GitHub Actions run','H2X'),Image(str(BASE/'screenshots/github-actions.png'),width=120*mm,height=98*mm),P('The public main-branch workflow for commit b0f1f75 completed successfully in 2 minutes 9 seconds. Screenshot captured inside the Prayogshala VM.','NoteX')]
story += [PageBreak()]
# 9 deploy
story += heading('Kubernetes deployment and monitoring','08 / Operations')
story += [P('The Kubernetes manifest defines a two-replica Deployment, a LoadBalancer Service on port 8000, readiness and liveness probes, resource requests and limits, and a non-root container security context. Prometheus scrape annotations point to /metrics. A Minikube deployment can expose the service through minikube tunnel; if tunnel cannot receive an external IP in the lab, port-forward is an honest local access path.')]
story += [table([['Check','Command / expected observation'],['Rollout','kubectl rollout status deployment/heartlab-api'],['Pods and service','kubectl get pods,service; two ready pods'],['Endpoint','curl /health and POST /predict through the exposed address'],['Logs','kubectl logs deployment/heartlab-api'],['Metrics','curl /metrics; request counter and latency histogram']], [46*mm,125*mm])]
story += [S(12),P('Monitoring interpretation','H2X'),P('The counter reveals successful, unavailable and failed requests. The latency histogram can show slowing predictions. These are operational metrics, not data-drift or clinical safety monitoring. A larger deployment should add availability alerts, distribution drift checks, retraining governance and review of model performance on newly labelled data.')]
story += [P('Observed lab deployment','H2X'),P('The first rollout failed because Kubernetes could not verify the image’s named non-root user. Adding runAsUser: 1000 to the manifest fixed it. The second rollout reported 2/2 replicas available, and both pods were ready. The LoadBalancer external IP stayed pending in Minikube, so I used port-forward for the service request. The direct lab screenshot is saved at screenshots/kubernetes.png, with a command log in reports/LAB_VERIFICATION.md.','NoteX')]
story += [PageBreak()]
# 10 reflection
story += heading('Verification, limitations and references','09 / Reflection')
status=[['Item','Current evidence'],['UCI data','Verified locally and in lab: 303 rows; cleaned CSV included'],['Code checks','Ruff passed; eight Pytest tests passed in lab'],['Lab dependencies','Installed in persistent ~/workspace/.venv'],['MLflow and model','Two candidate runs and selection run logged in lab; model saved'],['API','Container and Kubernetes service returned ready and prediction'],['Docker/Kubernetes','Image built; two ready pods; direct lab screenshot saved'],['GitHub Actions','Passed on first public push; screenshot included'],['Demonstration video','Recorded in the Prayogshala VM as WebM']]
story += [table(status,[53*mm,118*mm]),S(11)]
story += [P('Limitations','H2X'),P('The data is historical and limited to 303 Cleveland records. A single held-out split has high variance; cross-validation helps comparison but does not establish clinical transportability. The positive-class probability is not clinically calibrated. Missing values are imputed from training folds, but the training set is too small to guarantee stable estimates. The API is a teaching example and must not guide care.')]
story += [P('What I would improve next','H2X'),P('With more time and approved contemporary data, I would add a locked dependency environment, calibration assessment, repeated external validation, drift dashboards, CI image publication and a gated deployment workflow. I would also document how threshold choice changes precision and recall before any real-world use.')]
story += [P('References','H2X'),P('1. Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.<br/>2. BITS Pilani, AIMLCZG523 Assignment 01 instructions (supplied PDF).<br/>3. BITS Pilani, Prayogshala Virtual Lab Manual (supplied PDF).','SmallX')]

OUT.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=24*mm,bottomMargin=20*mm,title='MLOps Experimental Learning Assignment 01',author='Mahendra Singh')
doc.build(story,onFirstPage=page,onLaterPages=page)
print(OUT)
