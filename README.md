# 🔬 Tumour Risk Predictor

An interactive prediction dashboard that estimates whether a breast-mass cell nucleus is **benign** or **malignant** from eight numeric measurements, returning a probability, a risk meter and a per-feature explanation in real time. A batch mode scores whole CSV files.

**Live demo:** https://drive.google.com/file/d/1XX5nJab9Nggnl4h5p4LVU09IyG7jKvmt/view?usp=drivesdk

**Repository:** https://github.com/ansh25306/tumor-risk-predictor

**Web Link:** http://localhost:8501/
> ⚠️ Educational project. It is **not** a medical device and must not be used for diagnosis.

## Features

- **Single prediction form** with example profiles, per-field help text and the valid range shown as a placeholder.
- **Dynamic output:** colour-coded risk band (always paired with text), risk meter, class-probability bars, and a chart of which features pushed the prediction toward malignant or benign.
- **Batch mode:** upload a CSV, get a scored table and a downloadable results file. Bad rows are flagged with their spreadsheet row number instead of failing the whole file.
- **Graceful error handling:**
  - missing / blank fields → every problem listed at once
  - non-numeric, negative, NaN/inf or implausibly large values → rejected with a clear message
  - values outside the training range → prediction still returned, with a reliability warning
  - wrong file type, oversized file, empty file, malformed CSV, missing columns, too many rows → specific messages
  - missing or corrupt model files → readable error instead of a stack trace
- **Fast:** inference takes about a millisecond and is displayed with each result.

## Dataset and model

| | |
|---|---|
| Dataset | [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), UCI ML Repository (569 samples; bundled in scikit-learn as `load_breast_cancer`) |
| Features used | mean radius, texture, area, smoothness, compactness, concavity, concave points, symmetry |
| Target | 1 = malignant (212 samples), 0 = benign (357 samples) |
| Model | `StandardScaler` + `LogisticRegression` (C tuned by 5-fold grid search), saved with `joblib` |
| Selection | 5-fold CV ROC-AUC on the training split; a random forest was compared and only preferred if better by more than one standard deviation |

**Performance** (20% stratified held-out test set, 114 samples, touched once for reporting):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 92.1% | 86.7% | 92.9% | 89.7% | 0.985 |

Confusion matrix: 66 TN, 6 FP, 3 FN, 39 TP. Cross-validated ROC-AUC on the training split: logistic regression 0.985 ± 0.006, random forest 0.986 ± 0.007 (a tie within noise, so the simpler and more explainable model was used). The deployed model is refit on all 569 samples.

**Limitations:** small single-source dataset, summary measurements rather than images, and probabilities are model estimates rather than clinical risk.

## Tech stack

Python 3.13 · Streamlit · scikit-learn · pandas · NumPy · joblib (see [`requirements.txt`](requirements.txt))

## Project structure

```
├── app.py               # Streamlit UI
├── predictor.py         # model loading, validation, inference (no UI code)
├── train.py             # reproducible training script
├── artifacts/
│   ├── model.joblib     # serialized scikit-learn pipeline
│   └── metadata.json    # feature schema, ranges, examples, metrics
├── tests/test_predictor.py
├── requirements.txt
└── .streamlit/config.toml
```

## Run locally

```bash
git clone <your-repo-url>
cd tumor-risk-predictor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

### Reproduce training

```bash
python train.py
```

This re-creates `artifacts/model.joblib` and `artifacts/metadata.json` (seeded, so results are repeatable). The model is pinned to the scikit-learn version in `requirements.txt`; if you change that version, re-run `train.py`.

### Run the tests

```bash
python -m unittest discover -s tests -v
```

## Deploy

### Streamlit Community Cloud (recommended)

1. Push this repository to GitHub (public).
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python 3.13 (or the version you trained with) and deploy.
4. Paste the generated URL at the top of this README.

### Hugging Face Spaces (alternative)

Create a Space with the **Streamlit** SDK, upload the same files, and add this header at the top of `README.md`:

```yaml
---
title: Tumour Risk Predictor
sdk: streamlit
app_file: app.py
---
```
