# Sentiment Analysis

A modular machine learning pipeline for classifying Yelp restaurant reviews by star rating (1–5). The pipeline covers data loading, exploratory analysis, text preprocessing, and training of three progressively complex models.

## Project Structure

```
.
├── main.py
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── docs/
│   ├── main.md
│   ├── data_loader.md
│   ├── eda.md
│   ├── preprocessing.md
│   ├── models.md
│   ├── train.md
│   └── utils.md
├── data/
│   └── yelp_review/
│       ├── train.tsv
│       ├── val.tsv
│       └── test.tsv
├── model/
│   ├── best_cbow.pth
│   └── best_bilstm.pth
└── img/
    ├── rating_distribution.png
    └── review_length_distribution.png
```

## Models

| Model | Type | Library |
|---|---|---|
| Baseline | TF-IDF + Logistic Regression | scikit-learn |
| CBOW | Averaged word embeddings + FC layers | PyTorch |
| BiLSTM | Bidirectional LSTM | PyTorch |

All neural models use pretrained 300-dimensional word vectors from spaCy (`en_core_web_sm`).

## Usage

```bash
python3 main.py
```

The pipeline runs all stages automatically:
1. Downloads the dataset if not present
2. Runs EDA and saves plots to `img/`
3. Preprocesses text and builds DataLoaders
4. Trains Baseline → CBOW → BiLSTM in sequence
5. Saves best neural model checkpoints to `model/`

## Results

Best model: **BiLSTM** — saved at epoch 5 (early stopping triggered at epoch 7, patience=3).

| Model | Val Macro F1 |
|---|---|
| BiLSTM (best checkpoint) | **0.5310** |

Training log:

| Epoch | Train Loss | Val Loss | Val F1 |
|---|---|---|---|
| 1 | 1.4031 | 1.2327 | 0.4521 |
| 2 | 1.2334 | 1.2392 | 0.4346 |
| 3 | 1.1421 | 1.1874 | 0.4686 |
| 4 | 1.0361 | 1.0636 | 0.5289 |
| **5** | **0.9297** | **1.0446** | **0.5310** ← saved |
| 6 | 0.8368 | 1.0540 | 0.5501 |
| 7 | 0.7459 | 1.0646 | 0.5570 |

Checkpoint is saved based on lowest validation loss. Early stopping halted training after 3 consecutive epochs without improvement.

## Documentation

Per-module documentation is in [`docs/`](docs/). Each file covers function signatures, parameters, return values, and architecture details.

## Requirements

Install dependencies before running:

```bash
pip install -r requirements.txt
```
