# PlasticDeg Predictor — Complete Project Explanation

## AI-Based Prediction of Plastic-Degrading Potential in Bacteria and Fungi Using Genomic Data

---

## 1. Problem Statement

Plastic pollution is one of the most pressing environmental challenges. Certain bacteria and fungi produce enzymes that can **biologically degrade plastics** (e.g., PET, polyester). However, discovering these enzymes through lab experiments is slow and expensive.

This project uses **Machine Learning** to predict whether a given protein/enzyme sequence has plastic-degrading potential — enabling faster screening of candidate enzymes from genomic databases.

**Input**: A protein amino acid sequence (e.g., `MFPQSRHFGATV...`)
**Output**: Classification as **"Plastic-Degrading"** or **"Non-Degrading"** with a confidence score.

---

## 2. Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PlasticDeg Predictor                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │  Data     │──>│  Data    │──>│ Feature  │──>│ Model   │ │
│  │Collection │   │Preparing │   │Extraction│   │Training │ │
│  │ (UniProt) │   │& Cleaning│   │(K-mer /  │   │& Eval   │ │
│  │          │   │          │   │ ESM-2)   │   │         │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬────┘ │
│       Step 1        Step 2        Step 3/4       Step 5    │
│                                                     │      │
│                                              ┌──────▼─────┐│
│  ┌───────────┐       ┌──────────┐            │  Saved     ││
│  │ Next.js   │ <──── │ FastAPI  │ <───────── │  Models    ││
│  │ Frontend  │  HTTP │ Backend  │   joblib   │ (.joblib)  ││
│  │ (React)   │       │ (Python) │            │            ││
│  └───────────┘       └──────────┘            └────────────┘│
│   Port 3000          Port 8000                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. The ML Pipeline (Step-by-Step)

The entire pipeline is executed by running `python run_pipeline.py` and consists of **5 sequential steps**.

---

### Step 1 — Data Collection from UniProt

**File**: `src/data_collection/fetch_uniprot.py`

**What it does**: Downloads protein sequences from **UniProt**, the world's largest public protein sequence database, using its REST API.

**Positive Samples (Plastic-Degrading Enzymes)**:
| Enzyme Family | Role in Plastic Degradation |
|---|---|
| PETase | Breaks down PET plastic (polyethylene terephthalate) |
| MHETase | Degrades MHET (a PET breakdown product) |
| Cutinase (EC 3.1.1.74) | Hydrolyzes cutin and polyesters |
| Laccase (EC 1.10.3.2) | Oxidizes polyethylene and other plastics |
| Lipase (EC 3.1.1.3) | Hydrolyzes ester bonds in plastics |
| Mn Peroxidase (EC 1.11.1.13) | Degrades lignin and polyethylene |
| Lignin Peroxidase (EC 1.11.1.14) | Degrades lignin-like polymers |
| Polyester esterase | Breaks ester bonds in polyesters |

**Negative Samples (Non-Degrading Enzymes)**:
General metabolic enzymes that have no plastic-degrading function — alcohol dehydrogenase, hexokinase, DNA polymerase, RNA polymerase. These serve as the "control group".

**Output**: FASTA files saved in `data/raw/` (standard bioinformatics sequence format).

---

### Step 2 — Dataset Preparation

**File**: `src/data_collection/prepare_dataset.py`

**What it does**:
1. Reads all raw FASTA files
2. Assigns binary labels: **1 = plastic-degrading**, **0 = non-degrading**
3. Removes duplicate sequences (by UniProt ID)
4. Filters out sequences outside the valid length range (50–5000 amino acids)
5. Saves a clean CSV dataset

**Dataset Statistics**:
| Property | Value |
|---|---|
| Total samples | **907** |
| Positive (plastic-degrading) | **300** (33.1%) |
| Negative (non-degrading) | **607** (66.9%) |
| Average sequence length | **507 amino acids** |
| Min / Max length | 56 / 2894 amino acids |
| Enzyme families covered | PETase, MHETase, Cutinase, Laccase, Lipase, Mn Peroxidase, Lignin Peroxidase |

**Output**: `data/processed/dataset.csv` with columns: `id, description, sequence, length, label, source`

---

### Step 3 — Feature Engineering: K-mer Frequencies

**File**: `src/features/kmer_features.py`

**Why this is needed**: ML models cannot process raw text sequences directly. We need to convert amino acid sequences into **fixed-length numerical vectors**.

**Method**: Compute normalized frequencies of **k-mers** (subsequences of length k):

**1-mers (Amino Acid Composition)**:
- Count how often each of the 20 standard amino acids appears in the sequence
- Normalize by sequence length to get frequencies
- Produces **20 features** (one per amino acid: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)

**2-mers (Dipeptide Composition)**:
- Count every pair of consecutive amino acids (AA, AC, AD, ..., YW, YY)
- 20 x 20 = **400 possible dipeptides**
- Normalize by total number of dipeptides in the sequence
- Produces **400 features**

**Total feature vector**: 20 + 400 = **420 features per protein sequence**

**Example**:
```
Sequence: ACDEF...
1-mer "A" frequency = count("A") / len(sequence)
2-mer "AC" frequency = count("AC") / (len(sequence) - 1)
```

**Output**: `data/processed/features_kmer.npz` — a matrix of shape (907, 420)

---

### Step 4 — Feature Engineering: ESM-2 Protein Embeddings (Optional)

**File**: `src/features/esm_embeddings.py`

This step uses **transfer learning** with a pretrained deep learning model.

#### What is ESM-2?

**ESM-2 (Evolutionary Scale Modeling 2)** is a protein language model developed by **Meta AI Research (FAIR)**. It is a **transformer neural network** (same architecture family as GPT and BERT) but trained on protein sequences instead of natural language.

#### How ESM-2 was pretrained:

| Property | Details |
|---|---|
| Training data | ~65 million protein sequences from **UniRef50** database |
| Training objective | **Masked Language Modeling (MLM)** — randomly mask ~15% of amino acids and predict them from context |
| Architecture | Transformer encoder with self-attention |
| Model used in this project | `esm2_t6_8M_UR50D` (6 layers, 8M parameters, 320-dim output) |
| What it learns | Protein structure, folding patterns, evolutionary conservation, functional motifs |

#### How we use ESM-2 (Transfer Learning):

```
Protein Sequence (e.g., "MFPQSRHFGATV...")
        │
        ▼
┌──────────────────────┐
│  ESM-2 (pretrained,  │   ← Frozen weights, no retraining
│  frozen encoder)     │
└──────────┬───────────┘
           │
           ▼
  Per-token embeddings
  (one 320-dim vector per amino acid)
           │
           ▼
   Mean Pooling across
   sequence length
           │
           ▼
  Single 320-dim vector    ← This is the final embedding
  per protein
```

The model is **NOT retrained** on our data. We use it as a **frozen feature extractor** — it converts each protein into a rich 320-dimensional vector that captures structural and functional properties learned from 65 million proteins.

**K-mer vs ESM-2 Comparison**:
| Aspect | K-mer Features | ESM-2 Embeddings |
|---|---|---|
| Type | Handcrafted / Statistical | Learned / Deep Learning |
| Dimensions | 420 | 320 |
| Captures | Amino acid composition | Structure, function, evolution |
| Speed | Very fast (pure Python) | Slower (requires neural network inference) |
| GPU needed | No | Optional (CPU works for small model) |
| Accuracy | Good baseline | Generally higher |

**Output**: `data/embeddings/esm2_embeddings.npz` — a matrix of shape (907, 320)

---

### Step 5 — Model Training & Evaluation

**File**: `src/models/train.py`

Three ML models are trained and compared:

#### Model 1: Random Forest Classifier (on K-mer features)
- **Algorithm**: Ensemble of 200 decision trees
- **Input**: 420-dim k-mer feature vectors
- **Hyperparameters**: `n_estimators=200`, `max_depth=20`, `random_state=42`
- **Strengths**: Fast training, interpretable, robust to overfitting
- **Used for**: Real-time API predictions (default model)

#### Model 2: Support Vector Machine (on K-mer features)
- **Algorithm**: SVM with RBF (Radial Basis Function) kernel
- **Input**: 420-dim k-mer feature vectors
- **Hyperparameters**: `kernel="rbf"`, `probability=True`
- **Strengths**: Effective on small-to-medium datasets, good decision boundaries

#### Model 3: MLP Neural Network (on ESM-2 embeddings)
- **Algorithm**: Multi-Layer Perceptron (feedforward neural network)
- **Input**: 320-dim ESM-2 embedding vectors
- **Architecture**: 3 hidden layers (256 → 128 → 64 neurons), ReLU activation
- **Training**: Max 500 iterations, early stopping with 15% validation split
- **Strengths**: Best accuracy (leverages rich ESM-2 representations)

#### Preprocessing:
- **StandardScaler** is applied before training — scales features to zero mean and unit variance. This is critical for SVM and MLP which are sensitive to feature scales.

#### Evaluation Method: 5-Fold Stratified Cross-Validation
```
Full Dataset (907 samples)
    │
    ├── Fold 1: Train on 80% ──> Test on 20% ──> Metrics
    ├── Fold 2: Train on 80% ──> Test on 20% ──> Metrics
    ├── Fold 3: Train on 80% ──> Test on 20% ──> Metrics
    ├── Fold 4: Train on 80% ──> Test on 20% ──> Metrics
    └── Fold 5: Train on 80% ──> Test on 20% ──> Metrics
                                                    │
                                         Average ± Std Dev
```

- **Stratified** means each fold preserves the class ratio (33% positive / 67% negative)
- This prevents overfitting and gives a reliable estimate of generalization performance
- For each fold, a fresh scaler is fit on training data and applied to test data (no data leakage)

#### Evaluation Metrics:
| Metric | What it Measures |
|---|---|
| **Accuracy** | % of correct predictions overall |
| **Precision** | Of sequences predicted as plastic-degrading, what % actually are |
| **Recall** | Of all actual plastic-degrading sequences, what % did we find |
| **F1-Score** | Harmonic mean of precision and recall (balanced metric) |
| **AUC-ROC** | Area under the ROC curve — model's ability to distinguish classes across all thresholds |

#### Outputs:
- `src/models/saved/rf_kmer.joblib` — Saved Random Forest model + scaler
- `src/models/saved/mlp_esm.joblib` — Saved MLP model + scaler
- `src/models/saved/metrics.json` — All evaluation metrics
- `src/models/saved/roc_curves.png` — ROC curve comparison plot

---

## 4. Deployment

### Backend — FastAPI (Python)

**File**: `src/api/main.py`

A REST API that serves predictions in real time:

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Accepts a protein sequence string, returns prediction + confidence |
| `/predict/fasta` | POST | Accepts a FASTA file upload for batch prediction (up to 100 sequences) |
| `/health` | GET | Health check — lists loaded models |
| `/models` | GET | Returns available models and their evaluation metrics |

**Prediction flow** (when a user submits a sequence):
```
User submits sequence "MFPQSRH..."
        │
        ▼
  Input validation (length 20–5000)
        │
        ▼
  K-mer feature extraction (420-dim vector)
        │
        ▼
  StandardScaler transform (using saved scaler)
        │
        ▼
  Random Forest model.predict() + predict_proba()
        │
        ▼
  Response: { prediction: "Plastic-Degrading", confidence: 0.94, ... }
```

### Frontend — Next.js (React)

**File**: `frontend/app/page.js`

A web interface where users can:
1. Paste a protein sequence into a text box (or load an example PETase sequence)
2. Click "Predict" to send it to the FastAPI backend
3. View the result: prediction label, confidence percentage, probability breakdown
4. The UI uses a green/red badge to indicate the result visually

---

## 5. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Collection | UniProt REST API, BioPython, Requests | Fetch protein sequences |
| Data Processing | Pandas, NumPy | Dataset cleaning and manipulation |
| Feature Engineering | Custom k-mer code, Meta ESM-2 (PyTorch) | Convert sequences to numerical features |
| ML Models | Scikit-learn (RandomForest, SVM, MLP) | Classification |
| Evaluation | Scikit-learn metrics, Matplotlib, Seaborn | Cross-validation, ROC curves |
| Model Storage | Joblib | Serialize trained models |
| Backend API | FastAPI, Uvicorn | Serve predictions via REST API |
| Frontend | Next.js (React) | Web interface |

---

## 6. Key ML Concepts Used

1. **Binary Classification** — Two-class problem (plastic-degrading vs non-degrading)
2. **Feature Engineering** — Two approaches: handcrafted k-mer frequencies and learned ESM-2 embeddings
3. **Transfer Learning** — Using Meta's pretrained ESM-2 protein language model as a feature extractor without retraining it
4. **Masked Language Modeling** — The pretraining objective of ESM-2 (predict masked amino acids from context)
5. **Stratified K-Fold Cross-Validation** — Robust evaluation that avoids data leakage and preserves class ratios
6. **Feature Scaling (StandardScaler)** — Normalizing features to zero mean and unit variance before training
7. **Ensemble Methods** — Random Forest uses multiple decision trees and aggregates their votes
8. **Kernel Methods** — SVM with RBF kernel maps data to higher-dimensional space for better separation
9. **Neural Networks** — MLP with ReLU activation and early stopping for regularization
10. **Model Serialization** — Saving trained models with joblib for deployment
11. **ROC-AUC Analysis** — Evaluating classifier performance across all decision thresholds

---

## 7. File Structure

```
MINI PROJECT/
├── run_pipeline.py                  # Main pipeline runner (Steps 1-5)
├── requirements.txt                 # Python dependencies
│
├── data/
│   ├── raw/                         # Downloaded FASTA files from UniProt
│   │   ├── positive_petase.fasta    # PETase sequences
│   │   ├── positive_cutinase.fasta  # Cutinase sequences
│   │   ├── positive_laccase.fasta   # Laccase sequences
│   │   ├── positive_lipase.fasta    # Lipase sequences
│   │   ├── negative_*.fasta         # Non-degrading enzyme sequences
│   │   └── ...
│   ├── processed/
│   │   ├── dataset.csv              # Clean labeled dataset (907 samples)
│   │   ├── dataset_labeled.fasta    # Combined labeled FASTA
│   │   └── features_kmer.npz       # K-mer feature matrix (907 x 420)
│   └── embeddings/
│       └── esm2_embeddings.npz      # ESM-2 embedding matrix (907 x 320)
│
├── src/
│   ├── data_collection/
│   │   ├── fetch_uniprot.py         # Step 1: Download from UniProt API
│   │   └── prepare_dataset.py       # Step 2: Clean and label dataset
│   ├── features/
│   │   ├── kmer_features.py         # Step 3: K-mer frequency extraction
│   │   └── esm_embeddings.py        # Step 4: ESM-2 embedding extraction
│   ├── models/
│   │   ├── train.py                 # Step 5: Train RF, SVM, MLP
│   │   └── saved/                   # Saved models & metrics
│   └── api/
│       └── main.py                  # FastAPI backend server
│
└── frontend/
    └── app/
        └── page.js                  # Next.js React frontend
```

---

## 8. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full ML pipeline
python run_pipeline.py

# 3. Start the API server
uvicorn src.api.main:app --reload    # → http://localhost:8000

# 4. Start the frontend
cd frontend && npm install && npm run dev  # → http://localhost:3000
```

---

## 9. Summary

This project builds an **end-to-end ML system** that:
- Collects real biological data from UniProt
- Engineers features using both traditional (k-mer) and deep learning (ESM-2) approaches
- Trains and compares three ML models (Random Forest, SVM, MLP)
- Evaluates rigorously with cross-validation and multiple metrics
- Deploys as a web application with a FastAPI backend and Next.js frontend

The goal is to **accelerate the discovery of plastic-degrading enzymes** by computationally screening protein sequences before expensive lab experiments.
