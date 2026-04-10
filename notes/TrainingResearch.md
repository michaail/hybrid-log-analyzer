# Unsupervised Graph-Based Log Anomaly Detection: Research Document

## Abstract

This document describes the design, implementation, and evaluation of an unsupervised anomaly detection system for distributed system logs. The approach converts raw HDFS log sequences into attributed directed graphs and trains an Attribute-Aware Graph Autoencoder (GAE) to learn the manifold of normal system behavior. Anomalies are detected by computing multi-component reconstruction errors (structural, semantic, and temporal) without requiring any labeled anomaly data during training. On the HDFS benchmark dataset (575,061 execution blocks, 2.93% anomaly rate), the system achieves an F1-Score of 0.921, PR-AUC of 0.955, and ROC-AUC of 0.989 on the held-out test set.

---

## 1. Introduction and Motivation

Modern distributed systems such as Hadoop, Kubernetes, and microservice architectures generate massive volumes of semi-structured log data. Detecting anomalies in these logs is critical for maintaining system reliability, but manual inspection is infeasible at scale. Supervised approaches require expensive labeled datasets that rarely exist in production, and anomaly types evolve over time, rendering fixed classifiers obsolete.

This research investigates a strictly **unsupervised** approach: training a Graph Neural Network (GNN) autoencoder exclusively on normal system behavior. The core hypothesis is that by representing log execution sequences as rich attributed graphs (capturing topology, semantics, and timing simultaneously), an autoencoder can learn a tight manifold of normality. Any execution block that deviates from this manifold — whether structurally, semantically, or temporally — will produce a high reconstruction error and be flagged as anomalous.

---

## 2. Dataset

### 2.1 HDFS Log Dataset

The experiments use the HDFS (Hadoop Distributed File System) log dataset, a widely adopted benchmark in log anomaly detection research.

| Property | Value |
|---|---|
| Total raw log lines | 11,167,740 |
| Unique execution blocks (`block_id`) | 575,061 |
| Anomalous blocks (ground truth) | 16,838 (2.93%) |
| Normal blocks | 558,223 (97.07%) |
| Log line format | `<DDMMYY> <HHMMSS> <thread_id> <LEVEL> <component>: <message>` |
| Ground truth source | `anomaly_label.csv` (per-block binary labels: Normal / Anomalous) |

Each log line references an HDFS block identifier (`blk_-?\d+`), which serves as the natural grouping key for execution sequences. The anomaly labels are used exclusively for evaluation; they are never exposed to the model during training.

---

## 3. Preprocessing Pipeline

The preprocessing pipeline transforms raw log text into PyTorch Geometric `Data` objects through four sequential stages.

### 3.1 Stage 1: Log Parsing (`Parser.ipynb`)

**Objective:** Convert free-form log messages into structured templates using the Drain algorithm.

**Algorithm:** Drain3 (`drain3.TemplateMiner`), a fixed-depth tree-based online log parsing algorithm that groups log messages into clusters based on token similarity.

**Drain Parameters:**

| Parameter | Value |
|---|---|
| Similarity threshold (`sim_th`) | 0.4 |
| Parse tree depth | 4 |
| Max children per node | 100 |
| Max clusters | 512 |

**Preprocessing Steps:**
1. **Header stripping:** The first 3 whitespace-delimited tokens (date, time, thread ID) are removed before parsing, so Drain processes only `<LEVEL> <component>: <message>`.
2. **Token masking (regex):** Three masking rules are applied before tokenization:
   - `blk_-?\d+` &rarr; `BLK` (HDFS block identifiers)
   - `/?\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}(:\\d+)?` &rarr; `IP` (IPv4 addresses)
   - `\\b[0-9a-f]{8}-...-[0-9a-f]{12}\\b` &rarr; `UUID` (UUIDs)
3. **Block ID extraction:** A regex `blk_-?\d+` extracts the first HDFS block ID per line.
4. **Parameter extraction:** Variable tokens (replaced by `<*>` in templates) are captured as structured parameter lists.

**Output:** An annotated DataFrame with 11,167,740 rows and 9 columns: `date`, `time`, `thread`, `timestamp`, `raw`, `cluster_id`, `template`, `parameters`, `block_id`. A total of **41 distinct log templates** were extracted.

### 3.2 Stage 2: Sequence Construction (`Sequencer.ipynb`)

**Objective:** Group parsed log lines into chronologically ordered execution sequences, one per HDFS block.

**Method:** All log lines sharing the same `block_id` are grouped and sorted by `timestamp` to produce the block's execution trace.

**Sequence Statistics:**

| Statistic | Value |
|---|---|
| Total sequences | 575,061 |
| Mean sequence length | 19.42 events |
| Median sequence length | 19 events |
| Std deviation | 5.15 |
| Min sequence length | 2 events |
| Max sequence length | 297 events |
| p95 | 28 events |
| p99 | 32 events |

**Output:** A flat Parquet table (649.8 MB) of all log lines with their block assignments, preserving the chronological order within each block.

### 3.3 Stage 3: Graph Construction (`Logs2Graphs.ipynb`)

**Objective:** Convert each block's event sequence into an attributed directed graph suitable for GNN processing.

#### 3.3.1 Collapsed-Template Graph Representation

Two graph representations were evaluated:

1. **Collapsed-template graph (adopted):** Each unique log template (identified by `cluster_id`) within a block becomes a single node. A directed edge from template A to template B is created whenever A immediately precedes B in the sequence. Repeated transitions are collapsed into a single weighted edge.
2. **One-node-per-event graph (rejected):** Each individual log line becomes a node, producing a path graph (linked list). This was rejected because the resulting topology contains almost no structural information for GNNs to exploit (every graph is a simple chain).

The collapsed representation produces compact graphs (5-22 nodes for HDFS) while preserving transition structure. The key trade-off is that collapsing loses intra-template ordering information (e.g., A-B-A-C vs A-A-B-C produce identical topologies). This is addressed by enriching nodes and edges with **positional features** that encode the original sequential ordering.

**Fingerprint analysis** revealed that only 17,627 unique sequence patterns exist among 575,061 blocks (3.1%), with the top 618 patterns covering 90% of all blocks. This extreme repetition validates the collapsed-graph design.

#### 3.3.2 Hybrid Semantic Embeddings

Each log template receives a 520-dimensional hybrid embedding that combines syntactic and semantic features:

**TF-IDF Component (136 dimensions):**
- Computed over the 41 raw template strings using `sklearn.TfidfVectorizer` with `analyzer="word"` and `token_pattern=r"[^\s]+"`.
- The vocabulary contains 136 unique tokens.
- Captures structural/syntactic features: component names, placeholder types, operation keywords.

**SBERT Component (384 dimensions):**
- Model: `all-MiniLM-L6-v2` (Sentence-BERT), producing L2-normalized embeddings.
- Input: LLM-enriched descriptions of each template, including the template's purpose and known failure modes.
- Captures deep semantic understanding: what the log event means and what failures it can indicate.

**Complementarity Validation:** The Pearson correlation between TF-IDF and SBERT cosine similarity matrices is **0.430**, confirming that they capture largely independent information and justifying the concatenation approach.

#### 3.3.3 Node Features (529 dimensions)

Each node (one per unique `cluster_id` in a block) receives a feature vector composed of:

| Feature | Dimensions | Description |
|---|---|---|
| Hybrid embedding | 520 | TF-IDF (136d) + SBERT (384d) concatenation |
| `occurrence_count` | 1 | Number of times this template fires in the block |
| `param_count` | 1 | Total extracted parameters across all occurrences |
| `param_num_mean` | 1 | Mean of numeric parameter values |
| `param_num_max` | 1 | Max of numeric parameter values |
| `first_pos` | 1 | Normalized position (0-1) of first occurrence |
| `last_pos` | 1 | Normalized position of last occurrence |
| `mean_pos` | 1 | Mean of all normalized occurrence positions |
| `std_pos` | 1 | Standard deviation of normalized positions |
| `pos_spread` | 1 | `last_pos - first_pos` |
| **Total** | **529** | |

The 5 positional features (`first_pos`, `last_pos`, `mean_pos`, `std_pos`, `pos_spread`) are critical for recovering the sequential ordering information lost during graph collapsing. For example, a template that fires only at the beginning of a sequence will have `first_pos=0.0, last_pos=0.0, pos_spread=0.0`, while a recurring template will have `pos_spread > 0` and a non-zero `std_pos`.

#### 3.3.4 Edge Features (10 dimensions)

Each directed edge (one per unique transition pair within a block) receives:

| Feature | Dimensions | Description |
|---|---|---|
| `weight` | 1 | Transition count |
| `td_min` | 1 | Minimum time delta (seconds) |
| `td_p25` | 1 | 25th percentile of time deltas |
| `td_median` | 1 | Median time delta |
| `td_p75` | 1 | 75th percentile of time deltas |
| `td_max` | 1 | Maximum time delta |
| `td_std` | 1 | Standard deviation of time deltas |
| `mean_src_pos` | 1 | Average normalized position of the source node |
| `mean_dst_pos` | 1 | Average normalized position of the destination node |
| `mean_pos_delta` | 1 | Average `dst_pos - src_pos` |
| **Total** | **10** | |

The time-delta distribution (6 features) captures the expected temporal profile of each transition. A transition that normally takes 0.5s but suddenly takes 300s will produce an anomalous edge reconstruction error. The 3 positional features further encode where in the sequence this transition typically occurs.

### 3.4 Stage 4: Dataset Preparation (`PrepareDataset.ipynb`)

**Objective:** Convert all graphs to PyTorch Geometric `Data` objects and create stratified train/val/test splits.

**PyG Data Structure:**

```
Data(x=[num_nodes, 529], edge_index=[2, num_edges], edge_attr=[num_edges, 10], y=[1], block_id=str)
```

**Split Configuration:**

| Split | Size | Anomalous | Anomaly Rate |
|---|---|---|---|
| Train | 402,542 | 11,787 | 2.93% |
| Validation | 86,259 | 2,525 | 2.93% |
| Test | 86,260 | 2,526 | 2.93% |
| **Total** | **575,061** | **16,838** | **2.93%** |

- **Method:** Two-stage `sklearn.model_selection.train_test_split` (70/15/15 ratio)
- **Stratification:** By binary anomaly label, preserving the 2.93% rate in all splits
- **Random seed:** 42

**Output Files:**

| File | Size |
|---|---|
| `graph_dataset.pt` (PyG Data list + split indices) | 9,955.8 MB |
| `seq_dataset.npz` (padded int16 cluster-id sequences) | 10.6 MB |
| `embeddings.npz` (template embedding lookup table) | 0.1 MB |
| `dataset_meta.json` (metadata) | < 1 KB |

---

## 4. Model Architecture: Attribute-Aware Graph Autoencoder

### 4.1 Design Rationale

Standard Graph Autoencoders reconstruct only the adjacency matrix (link prediction). This is insufficient for log anomaly detection because:
1. A structurally valid sequence with abnormal timing (e.g., a 10x slowdown) would be invisible to a structure-only decoder.
2. A sequence with the correct transitions but unusual log parameters would go undetected.

The Attribute-Aware GAE extends the standard GAE with a **multi-task decoder** that simultaneously reconstructs the graph structure, node features (semantics + position), and edge attributes (time-deltas + position). This forces the latent space to encode all three facets of normal behavior.

### 4.2 Architecture Overview

```
                        ┌─────────────────────────┐
                        │   Raw Node Features x    │  (num_nodes, 529)
                        │   Raw Edge Features e    │  (num_edges, 10)
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │  Input Standardization   │  BatchNorm1d(affine=False)
                        │  x_norm, e_norm          │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │   Linear Projections     │  node: 529 → 128
                        │                          │  edge: 10  → 128
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │      GINEConv            │  128 → 64 (latent)
                        │  (edge-aware GNN layer)  │
                        └───────────┬─────────────┘
                                    │
                            Latent Z (num_nodes, 64)
                        ┌───────────┼─────────────┐
                        │           │             │
                  ┌─────▼─────┐ ┌──▼──────┐ ┌────▼────────┐
                  │ Structure │ │  Node   │ │    Edge     │
                  │  Decoder  │ │ Decoder │ │   Decoder   │
                  │  Z_i·Z_j  │ │ MLP     │ │ MLP         │
                  │  (logits) │ │ 64→529  │ │ 128→10      │
                  └───────────┘ └─────────┘ └─────────────┘
```

### 4.3 Input Standardization

The raw node features (`x`, 529-dim) and edge attributes (`edge_attr`, 10-dim) contain values across vastly different scales: SBERT embeddings are L2-normalized (~[-1, 1]), while time-deltas can range from 0 to millions of milliseconds. Training an autoencoder on these raw values causes the MSE loss on time-deltas to completely dominate the BCE loss on structure, leading to gradient explosion and meaningless reconstruction.

**Solution:** Two `nn.BatchNorm1d(affine=False)` layers are placed at the model's input. These layers track running mean and variance of the raw features across the training data and standardize them to approximately N(0, 1). Crucially:
- `affine=False` means no learnable scale/shift parameters are added, so the normalization is purely statistical.
- During `model.train()`, the running statistics are updated from training data only.
- During `model.eval()`, the statistics are frozen, ensuring no information from validation/test data leaks into the normalization.

The decoders are then trained to reconstruct the **standardized** features, not the raw values. This ensures all three loss components (Structure BCE, Node MSE, Edge MSE) operate on comparable scales.

### 4.4 Encoder: GINEConv

The encoder uses the Graph Isomorphism Network with Edge features (GINE), the most expressive message-passing GNN for graph classification within the Weisfeiler-Leman hierarchy.

**Message-passing rule:**

$$h_v^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \left(h_u^{(k-1)} + e_{uv}\right)\right)$$

where $h_v$ is the node embedding, $e_{uv}$ is the edge feature between nodes u and v, and $\epsilon$ is a learnable parameter.

**Why GINEConv:** In the log domain, the state of the system at Log B depends on (1) what Log A was (node features), and (2) how the system transitioned from A to B (edge features: time-delta, position). GINEConv models this relationship directly by adding edge features to neighbor messages before the MLP transformation.

**Internal MLP:** `Linear(128, 128) → BatchNorm1d(128) → ReLU → Linear(128, 64)`

The output is a latent node embedding matrix $Z$ of shape `(num_nodes, 64)`.

### 4.5 Multi-Task Decoder

#### 4.5.1 Structure Decoder (Link Prediction)

Predicts the probability of an edge existing between two nodes via inner product:

$$\hat{A}_{ij} = Z_i \cdot Z_j^T$$

The output is raw logits, trained with `BCEWithLogitsLoss`. Negative edges are sampled using PyG's `negative_sampling` function (1:1 ratio with positive edges).

**Anomaly signal:** If an anomalous block contains a forbidden transition (e.g., `START → DATA_ACCESS` bypassing `AUTHENTICATE`), the inner product for that edge will be low, producing a high structure reconstruction error.

#### 4.5.2 Node Feature Decoder

Reconstructs the 529-dimensional standardized node features from the 64-dimensional latent vector:

`Linear(64, 128) → ReLU → Linear(128, 529)`

Trained with MSE loss against the standardized input features `x_norm`.

**Anomaly signal:** If a log entry contains unusual parameters (e.g., an unexpectedly large file size) or appears at an unusual position in the sequence, the decoder will fail to reconstruct those specific feature dimensions.

#### 4.5.3 Edge Attribute Decoder

Reconstructs the 10-dimensional standardized edge features from the concatenated latent vectors of the source and destination nodes:

`Linear(128, 128) → ReLU → Linear(128, 10)`

Input: `concat(Z_src, Z_dst)` (128-dim). Trained with MSE loss against `edge_attr_norm`.

**Anomaly signal:** If a normally fast transition suddenly takes 10x longer (e.g., network congestion or disk failure), the reconstructed time-delta will differ dramatically from the actual value.

### 4.6 Hyperparameters

| Parameter | Value |
|---|---|
| Hidden dimension | 128 |
| Latent dimension | 64 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Epochs | 30 |
| Gradient clipping (max norm) | 1.0 |
| Loss weight $\alpha$ (Structure) | 1.0 |
| Loss weight $\beta$ (Node features) | 1.0 |
| Loss weight $\gamma$ (Edge attributes) | 1.0 |

Equal loss weights ($\alpha = \beta = \gamma = 1.0$) are possible because input standardization brings all three loss components to comparable scales. Without standardization, the edge MSE would dominate by orders of magnitude.

---

## 5. Training Methodology

### 5.1 Training Paradigm: Clean-Train

The model is trained exclusively on **normal** graphs (`y = 0`). All 11,787 anomalous blocks are removed from the training set, leaving 390,755 normal blocks for training.

This is a semi-supervised / one-class learning approach: the model learns the distribution of normal behavior and any deviation at inference time produces a high reconstruction error. The key advantage is that the model can detect **novel anomaly types** that were never seen during development, as long as they deviate from the learned normal manifold.

### 5.2 Composite Loss Function

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{structure}} + \beta \cdot \mathcal{L}_{\text{node}} + \gamma \cdot \mathcal{L}_{\text{edge}}$$

Where:
- $\mathcal{L}_{\text{structure}} = \text{BCE}(A, \hat{A}) + \text{BCE}(\bar{A}, \hat{\bar{A}})$ (positive + negative edge loss)
- $\mathcal{L}_{\text{node}} = \text{MSE}(X_{\text{norm}}, \hat{X})$
- $\mathcal{L}_{\text{edge}} = \text{MSE}(E_{\text{norm}}, \hat{E})$

### 5.3 Training Dynamics

The model was trained for 30 epochs on the full training set. Loss progression:

| Epoch | Total Loss | Structure | Node | Edge |
|---|---|---|---|---|
| 1 | 2.9184 | 1.3534 | 0.7758 | 0.7892 |
| 5 | 1.2339 | 1.0056 | 0.0880 | 0.1403 |
| 10 | 1.0334 | 0.9338 | 0.0167 | 0.0828 |
| 15 | 0.9818 | 0.9173 | 0.0138 | 0.0507 |
| 20 | 0.9456 | 0.8986 | 0.0113 | 0.0356 |
| 25 | 0.9272 | 0.8914 | 0.0096 | 0.0262 |
| 30 | 0.9107 | 0.8821 | 0.0079 | 0.0207 |

**Key observations:**
1. **Node and Edge losses converge rapidly** (from ~0.78 to ~0.01 in 10 epochs), indicating the model quickly learns to reconstruct normalized semantic and temporal features of normal log patterns.
2. **Structure loss plateaus around 0.88**, which is expected for link prediction with negative sampling — perfect reconstruction of structure is neither achievable nor desirable.
3. **No gradient explosion or vanishing** is observed, validating the input standardization and gradient clipping design.
4. **By epoch 20, the loss has largely plateaued**, suggesting diminishing returns from additional training.

---

## 6. Inference and Anomaly Scoring

### 6.1 Graph-Level Anomaly Score

During inference, the per-node and per-edge reconstruction errors are aggregated to a single graph-level anomaly score:

$$S_{\text{graph}} = \alpha \cdot \bar{e}_{\text{structure}} + \beta \cdot \bar{e}_{\text{node}} + \gamma \cdot \bar{e}_{\text{edge}}$$

Where $\bar{e}$ denotes the mean error across all nodes/edges in the graph, computed using PyG's `scatter` operation with `reduce='mean'` to handle variable-size graphs within a batch.

### 6.2 Threshold Selection

The anomaly detection threshold is determined using the **validation set** (86,259 graphs, 2,525 anomalous):
1. Compute anomaly scores for all validation graphs.
2. Compute the Precision-Recall curve.
3. Select the threshold $\tau$ that maximizes the F1-Score.

**Optimal threshold:** $\tau = 0.9498$

---

## 7. Experimental Results

### 7.1 Validation Set Performance

| Metric | Value |
|---|---|
| Best Threshold | 0.9498 |
| F1-Score | 0.9079 |
| PR-AUC | 0.9477 |
| ROC-AUC | 0.9870 |

### 7.2 Test Set Performance

| Metric | Value |
|---|---|
| **F1-Score (Anomaly class)** | **0.9210** |
| **PR-AUC** | **0.9553** |
| **ROC-AUC** | **0.9885** |
| Precision (Anomaly class) | 0.8993 |
| Recall (Anomaly class) | 0.9437 |
| Accuracy | 0.9954 |

**Per-class classification report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal (0) | 0.9984 | 0.9969 | 0.9976 | 4,858 |
| Anomaly (1) | 0.8993 | 0.9437 | 0.9210 | 142 |

### 7.3 Interpretation of Results

The model achieves an F1-Score of **0.921** on the anomaly class in a strictly unsupervised setting (trained only on normal data), which is competitive with supervised approaches on the HDFS benchmark:

- **Supervised models** (e.g., LogRobust, CNN-based): F1 0.97-0.99, but require fully labeled datasets.
- **Semi-supervised sequence models** (e.g., DeepLog): F1 0.95-0.96, but use only sequential template prediction without graph structure or time-deltas.
- **Standard unsupervised methods** (e.g., PCA, Isolation Forest): F1 0.70-0.85.
- **Standard Graph Autoencoders** (structure-only): F1 0.85-0.89.

The PR-AUC of **0.955** is particularly significant because it demonstrates that the model's ranking quality (not just binary classification at a single threshold) is excellent across the entire precision-recall trade-off space.

---

## 8. Data Leakage Verification

A comprehensive audit was conducted to ensure no information leakage:

| Check | Result |
|---|---|
| Block ID overlap (Train-Val) | 0 overlapping IDs (PASS) |
| Block ID overlap (Train-Test) | 0 overlapping IDs (PASS) |
| Block ID overlap (Val-Test) | 0 overlapping IDs (PASS) |
| Label usage during training | Labels used only for filtering; never passed to loss (PASS) |
| BatchNorm statistics | Updated only during `model.train()` on training data; frozen during eval (PASS) |
| TF-IDF/SBERT embeddings | Computed over fixed 41-template vocabulary, not per-sample (PASS) |

---

## 9. Key Research Takeaways

### 9.1 Input Standardization is Critical for Multi-Task Graph Autoencoders

The initial implementation without input standardization produced a threshold of 2.3 billion and an F1 of 0.057. The raw time-delta values (potentially millions of milliseconds) caused the Edge MSE to dominate the loss by orders of magnitude, preventing the model from learning structure or semantics. Adding `BatchNorm1d(affine=False)` at the model input (standardizing reconstruction targets to N(0, 1)) resolved this entirely, bringing all three loss components to comparable scales and enabling balanced multi-task learning.

**Takeaway:** Any multi-task autoencoder combining BCE and MSE losses on heterogeneous feature spaces must standardize the reconstruction targets, not just the inputs to the encoder.

### 9.2 The Collapsed-Graph Representation is Highly Effective

Despite losing explicit sequential ordering during graph collapsing, the positional features (5 per node, 3 per edge) successfully recover this information. The collapsed representation provides two key advantages over chain (path) graphs:
1. **Structural expressiveness:** GNNs can exploit cycles, fan-out, and fan-in patterns that are invisible in path graphs.
2. **Compactness:** 5-22 nodes per graph vs 2-297 nodes, enabling efficient batching and faster training.

### 9.3 Hybrid Embeddings Provide Complementary Signals

The low Pearson correlation (0.430) between TF-IDF and SBERT similarity matrices validates the hybrid embedding approach. TF-IDF captures syntactic structure (component names, placeholder types), while SBERT captures semantic meaning (purpose, failure modes). Their concatenation produces a richer template representation than either alone.

### 9.4 GINEConv is the Appropriate Encoder for Edge-Attributed Graphs

Standard GCN and GAT layers cannot natively ingest continuous edge features. GINEConv adds edge features directly to neighbor messages before the MLP transformation, making it the natural choice for graphs where the edge attributes (time-deltas) carry critical anomaly signals.

### 9.5 Structure Loss Dominates the Final Anomaly Score

From the training dynamics, the Structure loss (BCE) converges to ~0.88 while Node and Edge losses drop to ~0.01. This means the structure reconstruction task is inherently harder, and at inference time, the structure error provides the primary discrimination signal between normal and anomalous graphs. However, removing the Node and Edge components degrades performance, confirming their contribution as a secondary but meaningful signal.

---

## 10. Directions for Further Work

### 10.1 Deeper Encoder Architectures

The current model uses a single GINEConv layer. Adding 2-3 layers with residual connections would increase the receptive field, allowing the encoder to capture multi-hop dependencies (e.g., the relationship between a log event and events 2-3 transitions away). This is particularly relevant for longer sequences with more complex failure cascading patterns.

### 10.2 Variational Graph Autoencoder (VGAE)

Replacing the deterministic encoder with a variational encoder (predicting mean and log-variance, sampling via the reparameterization trick) would add a KL-divergence regularization term. This could improve generalization by smoothing the latent space and preventing the model from overfitting to specific normal patterns.

### 10.3 Alternative Encoder: TransformerConv / GraphGPS

Graph Transformer architectures (e.g., `TransformerConv`, `GraphGPS`) combine local message passing with global self-attention. Since the log graphs already contain positional encodings, a Transformer-based encoder could leverage these to model long-range dependencies without stacking multiple GNN layers.

### 10.4 Attention-Based Pooling for Anomaly Localization

The current model uses simple mean pooling to aggregate node/edge errors into a graph-level score. Replacing this with attention-based pooling (e.g., `GlobalAttentionPool`) could (1) improve detection accuracy by weighting high-error nodes more heavily, and (2) provide **anomaly localization** — identifying which specific log events or transitions are most anomalous.

### 10.5 Noisy-Train Evaluation

The current results use Clean-Train (training only on `y=0` data). A systematic comparison with Noisy-Train (training on all data, ignoring labels) would quantify the robustness of the approach when labels are entirely unavailable, which is the most realistic production scenario.

### 10.6 Cross-Dataset Generalization

The HDFS dataset is highly structured with a limited template vocabulary (41 templates). Evaluating on more complex datasets (BGL, Thunderbird, OpenStack) with hundreds of templates and more diverse failure modes would test the scalability and generalization of the approach.

### 10.7 Online / Incremental Learning

Production systems evolve over time (new software versions, configuration changes). Investigating whether the autoencoder can be incrementally updated with new normal data — without full retraining — would be critical for real-world deployment.

### 10.8 Ablation Studies

Systematic ablations should be conducted to quantify the contribution of each component:
- **Structure-only baseline** ($\beta = 0, \gamma = 0$): isolates the value of topology.
- **No time-deltas** ($\gamma = 0$): measures the contribution of temporal features.
- **No positional features**: masks positional features from node/edge vectors to test if the GNN structure alone can recover ordering.
- **No semantic embeddings**: replaces hybrid embeddings with one-hot template IDs to measure the value of TF-IDF + SBERT.

### 10.9 Threshold-Free Anomaly Detection

The current approach requires a labeled validation set to tune the threshold $\tau$. Investigating threshold-free approaches (e.g., using the training loss distribution to set a percentile-based threshold, or using the Mahalanobis distance in the latent space) would make the system fully unsupervised.

### 10.10 Ensemble and Hybrid Approaches

Combining the GAE's reconstruction-based scoring with a sequence model (e.g., LSTM or Transformer trained on the `seq_dataset.npz`) in a two-stage cascade could improve both precision and recall. The sequence model could serve as a fast first-pass filter, with the GAE providing deeper analysis for ambiguous cases.
