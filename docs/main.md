# main.py

Entry point for the sentiment analysis pipeline. Orchestrates all stages in sequence from data loading to model training.

## Usage

```bash
python3 main.py
```

## Pipeline Stages

### 1. Setup
Calls `utils.set_seed(572)` to fix random state across Python, NumPy, and PyTorch for reproducibility. Selects CUDA if available, otherwise CPU.

### 2. Data Loading
Downloads the Yelp review dataset from Google Drive if not already present, then loads `train.tsv`, `val.tsv`, and `test.tsv` as Pandas DataFrames.

### 3. EDA
Runs exploratory analysis on the training set and saves two plots to `img/`:
- `rating_distribution.png`
- `review_length_distribution.png`

### 4. Preprocessing
- Encodes star-rating labels (`1star`–`5star`) to integers via `LabelEncoder`.
- Builds a word-to-index vocabulary from training text.
- Constructs a 300-dimensional embedding matrix using spaCy pretrained vectors.
- Wraps train and validation sets in PyTorch `DataLoader` objects (batch size 64).

> Test set labels are unavailable; a dummy label (`1star`) is assigned so the pipeline runs without errors.

### 5. Baseline Model
Trains a TF-IDF + Logistic Regression pipeline and reports Macro F1 on the validation set.

### 6. CBOW Model
Trains a Continuous Bag-of-Words neural network for 3 epochs. Best checkpoint saved to `model/best_cbow.pth`.

| Hyperparameter | Value |
|---|---|
| Hidden size | 100 |
| Dropout | 0.5 |
| Optimizer | Adam, lr=0.01 |
| Epochs | 3 |

### 7. BiLSTM Model
Trains a Bidirectional LSTM for up to 10 epochs with early stopping (patience=3). Best checkpoint saved to `model/best_bilstm.pth`.

| Hyperparameter | Value |
|---|---|
| Hidden size | 256 |
| Layers | 3 |
| Dropout | 0.5 |
| Optimizer | Adam, lr=0.001 |
| Epochs | 10 (max) |
| Early stopping patience | 3 |

## Dependencies
All `src/` modules: `utils`, `data_loader`, `eda`, `preprocessing`, `models`, `train`.
