# CI/CD Pipeline — Step-by-Step Explanation

This document explains every component of the CI/CD pipeline, what it does, and why it exists.

---

## Architecture Overview

The pipeline is composed of **5 independent GitHub Actions workflows**, a **DVC reproducible pipeline** for ML stages, **Papermill** for notebook execution, and **CML** for PR-based reporting. All of these are glued together by a common project structure and shared configuration.

```
Developer pushes code
        │
        ├──→ lint.yaml        (every push/PR — seconds)
        ├──→ test.yaml        (every push/PR — ~1 min)
        ├──→ evaluate.yaml    (PRs to main — ~5 min)
        ├──→ train.yaml       (merge to main — ~10 min)
        └──→ retrain.yaml     (weekly cron — ~10 min)
```

---

## Project Structure

| Path | Purpose |
|---|---|
| `src/` | Python source modules (preprocessing, model, evaluation, utilities) |
| `notebooks/` | Jupyter notebooks executed by Papermill (training, evaluation, LLM) |
| `tests/` | Pytest unit tests for all source modules |
| `configs/params.yaml` | Centralized hyperparameters consumed by DVC and Papermill |
| `dvc.yaml` | DVC pipeline DAG defining reproducible ML stages |
| `data/raw/` | Raw input log data (tracked by DVC) |
| `data/processed/` | Preprocessed graph data (DVC output) |
| `models/` | Trained model checkpoints (DVC output) |
| `plots/` | Generated evaluation plots (ROC, loss curves) |
| `outputs/` | Executed notebook outputs from Papermill (gitignored) |
| `metrics.json` | Evaluation metrics consumed by DVC and CML |
| `.github/workflows/` | GitHub Actions workflow definitions |

---

## Pipeline Stages (DVC)

The ML pipeline is defined in `dvc.yaml` as a directed acyclic graph (DAG) with three stages. DVC tracks dependencies and outputs so stages only re-run when their inputs change.

### Stage 1: `preprocess`

**Command:** `python src/preprocessing.py`

**What it does:**
- Reads raw log files (CSV/JSON) from `data/raw/`
- Parses logs into structured events
- Builds graph representations (nodes = log events, edges = relationships)
- Writes processed graph data to `data/processed/`

**Why:**
Raw logs must be transformed into graph-structured data before the GNN can consume them. This stage ensures preprocessing is reproducible and cached — it only re-runs when raw data or the preprocessing script changes.

**Dependencies:** `src/preprocessing.py`, `data/raw/`
**Outputs:** `data/processed/`

---

### Stage 2: `train`

**Command:** `papermill notebooks/train_gat.ipynb outputs/train_output.ipynb -f configs/params.yaml -k python3 --log-output`

**What it does:**
- Executes the training notebook programmatically via Papermill
- Injects hyperparameters from `configs/params.yaml` (learning rate, attention heads, batch size, etc.)
- Trains a Graph Attention Network (GAT) based autoencoder on the processed graph data
- Saves model checkpoint (weights + optimizer state) to `models/gat_model.pt`
- Includes early stopping and periodic checkpointing

**Why:**
- **Papermill** enables parameterized, headless notebook execution — the same notebook runs interactively in Colab (with GPU) and automatically in CI (CPU)
- **Environment guards** (`is_ci()`, `is_colab()`) ensure Colab-specific code (like `drive.mount()`) is skipped in CI
- **DVC dependency tracking** means training only re-runs when the notebook, data, params, or model code changes

**Dependencies:** `notebooks/train_gat.ipynb`, `data/processed/`, `configs/params.yaml`, `src/model.py`, `src/utils.py`
**Outputs:** `models/gat_model.pt`

---

### Stage 3: `evaluate`

**Command:** `papermill notebooks/evaluate.ipynb outputs/evaluate_output.ipynb -k python3 --log-output`

**What it does:**
- Loads the trained model checkpoint
- Runs inference on test data
- Computes metrics: AUROC, AUPRC, F1 score
- Generates plots: ROC curve, training loss curve
- Writes `metrics.json` (consumed by DVC metrics tracking and CML reporting)

**Why:**
Evaluation is decoupled from training so it can run independently (e.g., when only the evaluation logic changes). The metrics file is tracked by DVC with `cache: false` so it's always committed to Git — enabling `dvc metrics diff` comparisons across branches.

**Dependencies:** `notebooks/evaluate.ipynb`, `models/gat_model.pt`, `data/processed/`, `src/evaluate.py`, `src/utils.py`
**Metrics:** `metrics.json`
**Plots:** `plots/roc.png`, `plots/loss.png`

---

## GitHub Actions Workflows

### 1. `lint.yaml` — Code Quality

**Trigger:** Every push and pull request
**Runtime:** ~15 seconds

**Steps and rationale:**
1. **Ruff lint** (`ruff check src/ tests/`): Fast Python linter that catches syntax errors, unused imports, style violations. Catches issues before they reach code review.
2. **Ruff format check** (`ruff format --check`): Ensures consistent code formatting across all contributors.
3. **Mypy** (`mypy src/`): Static type checking catches type errors at CI time rather than runtime.
4. **nbstripout check**: Verifies notebook outputs have been stripped before commit. Notebook outputs bloat git history, cause merge conflicts, and may contain sensitive data.

**Why separate workflow:** Linting is fast and should never be blocked by slow training jobs. Developers get instant feedback.

---

### 2. `test.yaml` — Unit Tests

**Trigger:** Every push and pull request
**Runtime:** ~1 minute

**Steps and rationale:**
1. **Install dependencies**: Full `requirements.txt` so tests can import `torch`, `sklearn`, etc.
2. **Pytest** (`pytest tests/ -v --tb=short`): Runs unit tests for:
   - `test_preprocessing.py` — data loading, graph construction, serialization
   - `test_model.py` — checkpoint save/load round-trip
   - `test_evaluate.py` — metrics computation, JSON output
   - `test_utils.py` — environment detection, seeding, device selection

**Why:** Unit tests catch regressions in core logic without needing GPUs, data, or model artifacts. They run on every push for immediate feedback.

---

### 3. `evaluate.yaml` — PR Evaluation with CML

**Trigger:** Pull requests targeting `main`
**Runtime:** ~5 minutes

**Steps and rationale:**
1. **Full git history** (`fetch-depth: 0`): Required for `dvc metrics diff` to compare against the `main` branch.
2. **GCS authentication**: Service account credentials unlock the DVC remote to pull data and model artifacts.
3. **`dvc pull`**: Downloads datasets and trained models from Google Cloud Storage.
4. **`dvc repro evaluate`**: Runs only the evaluation stage (training is cached/skipped if the model hasn't changed).
5. **CML report generation**:
   - `dvc metrics diff main --show-md` → Markdown table comparing metrics between the PR branch and main
   - `cml asset publish plots/*.png` → Uploads plots as image assets
   - `cml comment create report.md` → Posts the full report as a PR comment

**Why:** Reviewers see the impact of code changes on model performance directly in the PR — no need to check external dashboards, run notebooks manually, or ask "did you test this?". The comparison against `main` makes regressions immediately visible.

---

### 4. `train.yaml` — Full Pipeline on Main

**Trigger:** Push to `main` (when notebook/src/config/dvc files change) + manual `workflow_dispatch`
**Runtime:** ~10 minutes (CPU)

**Steps and rationale:**
1. **DVC cache** (`actions/cache@v4`): Caches DVC data locally across runs to avoid re-downloading unchanged datasets.
2. **Parameter override from dispatch**: When triggered manually via `workflow_dispatch`, allows overriding learning rate, epochs, and attention heads without editing `params.yaml`.
3. **`dvc repro`**: Runs the full pipeline (preprocess → train → evaluate). DVC skips stages whose inputs haven't changed.
4. **`dvc push`**: Uploads new model artifacts and cache back to the GCS remote.
5. **Auto-commit**: Updates `dvc.lock` and `metrics.json` in Git so the repo always reflects the latest pipeline state. Uses `[skip ci]` in the commit message to prevent infinite CI loops.

**Why:** Every merge to `main` ensures the pipeline is reproducible and artifacts are synced. The `workflow_dispatch` input parameters enable quick experiment iteration without code changes.

**CPU-only note:** In CI, training runs on a CPU with a small data subset (or uses a pre-existing model from Colab). Full GPU training happens interactively in Google Colab — the developer runs `dvc push` after training and commits `dvc.lock`.

---

### 5. `retrain.yaml` — Scheduled Retraining

**Trigger:** Weekly cron (Sunday 2:00 AM UTC) + manual dispatch
**Runtime:** ~10 minutes

**Steps and rationale:**
- Identical to `train.yaml` but triggered on a schedule
- Ensures the model stays up-to-date with any new data that arrives between manual retraining sessions
- Auto-commits updated `dvc.lock` and `metrics.json`

**Why:** Anomaly detection models can drift as log patterns evolve. Weekly retraining provides a safety net even when developers aren't actively experimenting. The `workflow_dispatch` fallback allows manual triggering after a data update.

---

## Key Tools and Why They're Used

### DVC (Data Version Control)
**What:** Git-like version control for data files and ML pipelines.
**Why:** Git can't handle large binary files (datasets, model weights). DVC stores them externally (GCS) while keeping lightweight pointer files (`.dvc`, `dvc.lock`) in Git. The `dvc.yaml` pipeline DAG ensures reproducibility — any contributor can run `dvc repro` and get the same results.

### Papermill
**What:** Programmatic execution of Jupyter notebooks with parameter injection.
**Why:** The project uses notebooks for training (researcher-friendly, visual). Papermill bridges the gap between interactive Colab development and headless CI execution. Parameters are injected from `configs/params.yaml` without modifying the notebook.

### CML (Continuous Machine Learning)
**What:** Posts ML metrics, plots, and reports as comments on GitHub PRs.
**Why:** Brings ML experiment results into the code review workflow. Reviewers see AUROC, F1, ROC curves, and loss graphs directly on the PR — no context switching to external tools.

### Google Cloud Storage (GCS)
**What:** Object storage for DVC remote (datasets, models, cache).
**Why chosen over Google Drive:** GCS has native service account authentication (no interactive OAuth), no quota limits, and first-class DVC support. Google Drive's `drive.mount()` requires interactive login and doesn't work in CI.

### nbstripout
**What:** Strips output cells from Jupyter notebooks before git commit.
**Why:** Notebook outputs (images, tensors, logs) bloat the repository, cause merge conflicts, and may leak sensitive data. The pre-commit hook strips them automatically.

### Pre-commit
**What:** Git hook framework that runs checks before each commit.
**Why:** Catches formatting, linting, and notebook output issues locally before they reach CI — faster feedback loop for the developer.

---

## Colab ↔ CI Synchronization

Since GPU training happens in Google Colab and CI is CPU-only, the workflow is:

```
1. Developer works in Colab (GPU training)
       ↓
2. Model saved → GCS via DVC (dvc push)
       ↓
3. Updated dvc.lock committed to Git
       ↓
4. PR opened → evaluate.yaml runs
       ↓
5. CI pulls model via dvc pull, runs evaluation
       ↓
6. CML posts metrics comparison on PR
       ↓
7. Reviewer approves, merges to main
```

The notebooks contain environment guards:
```python
if is_colab():
    drive.mount("/content/drive")  # Only in Colab
# In CI, local paths are used instead
```

---

## Secrets Required in GitHub

| Secret | Purpose | How to obtain |
|---|---|---|
| `GCP_SA_JSON` | Service account JSON key for GCS access (DVC remote, auth) | GCP Console → IAM → Service Accounts → Create key |
| `OPENAI_API_KEY` | OpenAI API access for LLM evaluation (optional) | platform.openai.com |
| `GEMINI_API_KEY` | Google Gemini API access (optional) | aistudio.google.com |

`GITHUB_TOKEN` is automatically provided by GitHub Actions — no setup needed. It's used by CML to post PR comments.

---

## Configuration

All hyperparameters live in `configs/params.yaml`:

```yaml
learning_rate: 0.001
GAT_attention_heads: 4
batch_size: 32
epochs: 50
# ... etc
```

This single file is:
- A **DVC dependency** (training re-runs when params change)
- Consumed by **Papermill** (`-f configs/params.yaml`)
- Overridable via **`workflow_dispatch` inputs** for quick experiments
