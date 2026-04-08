# Notebooks Walkthrough & Pipeline Documentation

This directory contains the core Jupyter notebooks used for log parsing, dataset preparation, and graph construction. These notebooks represent the sequential data preparation pipeline for the Graph Anomaly Detection model.

## 1. Parser (`Parser.ipynb`)

**Objective:** Parse raw HDFS log files into structured templates and annotations using the Drain parsing algorithm.

**Data Inputs & Outputs:**
*   **Input:** `data/raw/HDFS_full.log` (Raw text log file)
*   **Outputs:** 
    *   `data/processed/{RUN_TAG}_hdfs_templates.json` (Parsed log templates)
    *   `data/processed/{RUN_TAG}_hdfs_annotated.parquet` (Structured DataFrame of log events with cluster IDs, timestamps, and parameters)
    *   `models/{RUN_TAG}_drain_parser.bin` (Fitted parser model)

**Main Components & Critical Sections:**
*   **Initialization & Tagging:** Uses a `RUN_TAG` (e.g., `20260407_1030`) for versioning outputs.
*   **Parsing (DrainParser):** Fits the Drain parser on the raw log. It identifies log templates (e.g., `Receiving block <BLK> src: <IP> dest: <IP>`).
*   **Annotation:** `parser.annotate_file` performs a second pass over the log, attributing each line to its corresponding template, cluster ID, and extracted parameters.
*   **Persistence:** The annotated logs are saved as a Parquet file. The script uses `json.dumps` to serialize the `parameters` list into strings, and casts object columns to plain python objects to bypass a `fastparquet` bug with Arrow-backed strings.

**Edge Cases & Suggested Fixes:**
*   *Issue:* Fastparquet/Pyarrow type conflict when serializing `ArrowDtype` strings.
*   *Fix:* Instead of coercing types to `object` to appease `fastparquet`, upgrade pandas/pyarrow and strictly use the `pyarrow` engine, which has native support for nested lists and string types, obviating the need for `json.dumps` on parameters.

---

## 2. EnrichTemplates (`EnrichTemplates.ipynb`)

**Objective:** Enhance parsed log templates with semantic meaning (purpose, failure modes) using Large Language Models (LLMs).

**Data Inputs & Outputs:**
*   **Input:** `data/processed/{RUN_TAG}_hdfs_templates.json`
*   **Output:** `data/processed/{RUN_TAG}_hdfs_templates_enriched.json`

**Main Components & Critical Sections:**
*   **Loading:** Automatically discovers the newest template JSON using `glob`.
*   **LLM Enrichment:** Utilizes `src.enricher.Enricher` (which calls an LLM like Mistral Large/Small via Azure OpenAI) to generate semantic context for each raw template.
*   **Merging:** Iterates through all templates, invokes the LLM, and attaches the resulting `enriched_large` and `enriched_small` JSON payloads to the templates.

**Data Provenance & Security:**
*   Relies on `.env` files for API Keys. 
*   *Security Note:* Ensure `.env` is listed in `.gitignore` to prevent credential leaks. Use GitHub Secrets (`AZURE_OPENAI_KEY`, etc.) in CI/CD.

**Edge Cases & Suggested Fixes:**
*   *Issue:* Sequential LLM calls in a `for` loop are slow and susceptible to transient network failures.
*   *Fix:* Implement asynchronous requests (e.g., `asyncio.gather`) with exponential backoff and retry logic (e.g., using the `tenacity` library) to ensure robustness against rate-limiting and timeouts.

---

## 3. Sequencer (`Sequencer.ipynb`)

**Objective:** Group chronologically ordered log events into distinct sequences based on their `block_id`.

**Data Inputs & Outputs:**
*   **Input:** `data/processed/{RUN_TAG}_hdfs_annotated.parquet`
*   **Output:** `data/processed/{RUN_TAG}_hdfs_sequences.parquet`

**Main Components & Critical Sections:**
*   **Grouping:** Filters out log lines without a `block_id` (e.g., NameNode bookkeeping) and sorts the remaining lines by `(block_id, timestamp)`. 
*   **Dictionary Generation:** Converts the grouped pandas DataFrame into a dictionary of DataFrames using `dict(df_blocks.groupby("block_id", sort=False))`. This avoids repeated `reset_index` allocations, speeding up execution.
*   **Sequence Stats:** Computes sequence length distributions and visualizes them.
*   **Persistence:** Dumps the sequence-ordered Parquet file for downstream consumption.

**Edge Cases & Suggested Fixes:**
*   *Issue:* Memory spikes when constructing the `sequences` dictionary for millions of blocks.
*   *Fix:* Refactor to group and stream directly to disk without keeping all sequences in RAM simultaneously, or use `dask.dataframe` for out-of-core processing.

---

## 4. Logs2Graphs (`Logs2Graphs.ipynb`)

**Objective:** Explore graph representations for HDFS log sequences and construct template embeddings.

**Data Inputs & Outputs:**
*   **Inputs:** `{RUN_TAG}_hdfs_sequences.parquet`, `{RUN_TAG}_hdfs_templates_enriched.json`
*   **Outputs:** Exploratory visualizations and comparisons (No direct disk output, serves as a study for `PrepareDataset.ipynb`).

**Main Components & Critical Sections:**
*   **Node Representation Comparison:** Evaluates two approaches:
    *   *Collapsed-Template:* 1 node per template type. Edges represent transitions. Requires positional encodings (first, last, mean positions) to recover sequence ordering.
    *   *One-Node-per-Event:* 1 node per log line. Creates a linear chain (path graph).
    *   *Decision:* The Collapsed-Template approach is chosen because it creates structured topology suitable for Graph Neural Networks.
*   **Embeddings:** Compares TF-IDF (structural token features) against Sentence-BERT (`all-MiniLM-L6-v2`) on enriched semantic text. It proposes a **Hybrid approach** (TF-IDF + SBERT).
*   **Sequence Fingerprinting:** Hashes the ordered `cluster_id` sequences using MD5. Deduplicates structurally identical blocks (saves 90% computation time since most blocks follow standard patterns).

**Edge Cases & Suggested Fixes:**
*   *Issue:* The notebook dynamically installs dependencies via `!pip3 install torch==2.2.2 sentence-transformers==2.2.2`. This is an anti-pattern for reproducible pipelines.
*   *Fix:* Move all dependencies to `requirements.txt` or `pyproject.toml`.

---

## 5. PrepareDataset (`PrepareDataset.ipynb`)

**Objective:** Consolidate data into final PyTorch Geometric (`PyG`) graph objects and compact sequence matrices, ready for model training, and perform stratified data splitting.

**Data Inputs & Outputs:**
*   **Inputs:** `*_hdfs_sequences.parquet`, `*_hdfs_templates_enriched.json`, `data/raw/anomaly_label.csv`
*   **Outputs:** 
    *   `{RUN_TAG}_graph_dataset.pt` (PyG Data list and split indices)
    *   `{RUN_TAG}_seq_dataset.npz` (Int16 padded sequences for LSTM baselines)
    *   `{RUN_TAG}_embeddings.npz` (Hybrid embedding lookup table)
    *   `{RUN_TAG}_dataset_meta.json` (Splits, counts, dimensionality)

**Main Components & Critical Sections:**
*   **Label Loading:** Reads ground truth from `anomaly_label.csv`. Includes a fallback heuristic (tags any sequence containing "ERROR/WARN" as an anomaly) if the CSV is missing.
*   **Embedding Construction:** Builds the TF-IDF + SBERT hybrid embeddings matrix.
*   **Graph Construction (`seq_to_pyg`):** Iterates over blocks, builds collapsed graphs, computes 10-dimensional edge features (time-delta stats + positional deltas) and node features (embeddings + positional stats), and instantiates PyG `Data` objects.
*   **Padding & Sequence Matrix:** Creates memory-efficient `int16` padded sequence matrices (`N, MAX_SEQ_LEN`) for sequence-first model baselines.
*   **Splitting:** Performs a 70/15/15 train/val/test stratified split based on anomaly labels to preserve the minority class ratio (~2.9% anomalies).

**Edge Cases & Suggested Fixes:**
*   *Issue:* Using `np.savez_compressed` with `allow_pickle=True` can be a security vulnerability and can break across Python/NumPy versions.
*   *Fix:* Use `h5py` for arrays and dictionary-based metadata, or store tabular sequence formats in Parquet. 

---

## Reproducing the Results

To reproduce these results in a clean environment, follow these steps:

### 1. Environment Setup

Ensure you have Python 3.10+ installed.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (specific versions based on notebook requirements)
pip install pandas fastparquet scikit-learn networkx matplotlib python-dotenv tqdm jupyter papermill
pip install torch==2.2.2 sentence-transformers==2.2.2 torch_geometric
```

### 2. External Data & Authentication

1.  **HDFS Dataset & Labels:** 
    Download `HDFS_full.log` and `anomaly_label.csv` from the [LogHub Repository](https://github.com/logpai/loghub/tree/master/HDFS). Place them in the `data/raw/` directory.
2.  **LLM Enrichment (Optional but recommended):** 
    Create a `.env` file at the root of the project with your API keys:
    ```ini
    AZURE_OPENAI_DEPLOYMENT_MISTRAL_LARGE=your-deployment-name
    AZURE_OPENAI_KEY=your-api-key
    ```

### 3. Execution Pipeline

You can run the notebooks sequentially using `papermill` to execute them headlessly and reproducibly:

```bash
# 1. Parse Logs
papermill notebooks/Parser.ipynb outputs/Parser_out.ipynb

# 2. Enrich Templates
papermill notebooks/EnrichTemplates.ipynb outputs/Enrich_out.ipynb

# 3. Build Sequences
papermill notebooks/Sequencer.ipynb outputs/Sequencer_out.ipynb

# 4. Prepare PyG Dataset
papermill notebooks/PrepareDataset.ipynb outputs/PrepareDataset_out.ipynb
```

**Verification:**
After running the pipeline, inspect the `data/processed/` directory. You should see `.parquet` files, `.json` templates, `.pt` graph datasets, and `.npz` sequence arrays bearing the same `YYYYMMDD_HHMM` run tag. Check the `{RUN_TAG}_dataset_meta.json` to verify that the split ratios are correct and that the `anomaly_rate` roughly matches ~2.9%.

---

## Modularity, Robustness, & Refactoring Plan

The current workflow relies on chained Jupyter notebooks. While great for exploration, they are fragile for CI/CD ML pipelines (state leakages, implicit execution order, hardcoded paths).

**Identified Issues:**
1. **Dynamic Pathing & Tagging:** The notebooks search for `candidates[-1]` using `glob` to find the most recent previous output. This is fragile if old, corrupted runs exist in the directory.
2. **Duplicated Logic:** Embedding generation and graph construction logic is duplicated between `Logs2Graphs.ipynb` and `PrepareDataset.ipynb`.
3. **Data Serialization:** `fastparquet` workarounds (coercing types, json dumping lists) compromise performance. 

**Consolidation Plan (Towards a Coherent Workflow):**

We will refactor the pipeline into a unified, modular Python CLI application, structured via DVC (Data Version Control).

1.  **Extract Core Logic to `src/`:**
    *   Move `seq_to_pyg()` and embedding routines into `src/preprocessing.py`.
    *   Move MD5 fingerprinting and template collapsing into `src/graph_builder.py`.
2.  **Explicit DVC Pipeline:**
    *   Instead of auto-picking timestamps, DVC will handle inputs/outputs with explicit, deterministic filenames (e.g., `data/processed/hdfs_sequences.parquet`). The timestamping logic will be moved to the final artifact registry, not the intermediate steps.
3.  **Use PyArrow:**
    *   Switch to PyArrow natively: `df.to_parquet("...", engine="pyarrow")`. Remove `json.dumps()` hacks.
4.  **Logging & Testing:**
    *   Replace `print()` statements with standard `logging`.
    *   Add PyTest cases for `seq_to_pyg()` edge cases (e.g., blocks with single events, blocks with no numeric parameters).

### Validation Checklist
- [ ] Migrate `Parser.ipynb` logic to `src/parser_cli.py`.
- [ ] Replace `fastparquet` hacks with `pyarrow`.
- [ ] Move LLM enrichment to an async robust class in `src/enricher.py`.
- [ ] Centralize Graph and PyG dataset construction into `src/dataset_builder.py`.
- [ ] Define the end-to-end DAG in `dvc.yaml`.
- [ ] Verify `metrics.json` and split distributions remain identical to notebook baseline.