# src/models.py

Defines the three sentiment classification models: a scikit-learn baseline and two neural networks.

## Classes

### `BaselineModel`

A scikit-learn pipeline combining TF-IDF vectorization with Logistic Regression.

**Pipeline stages**
1. `TfidfVectorizer` — removes English stopwords, uses unigrams and bigrams (`ngram_range=(1,2)`).
2. `LogisticRegression` — `liblinear` solver, regularization `C=1.0`.

#### Methods

| Method | Signature | Description |
|---|---|---|
| `train` | `(X_train, y_train)` | Fits the pipeline on raw text and string labels. |
| `evaluate` | `(X_dev, y_dev)` | Predicts and prints Macro F1 score. Returns the F1 value. |

---

### `CBOW(nn.Module)`

Continuous Bag-of-Words classifier. Averages token embeddings over a sequence (excluding padding), then passes through two fully connected layers.

**Constructor parameters**
| Parameter | Type | Description |
|---|---|---|
| `weights_matrix` | `Tensor` | Pretrained embedding matrix `(vocab_size, emb_dim)`. |
| `num_classes` | `int` | Number of output classes. |
| `dropout_prob` | `float` | Dropout rate applied between the two linear layers. |
| `padding_idx` | `int` | Index of the `<PAD>` token; excluded from the mean. |
| `hidden_size` | `int` | Hidden layer size. Default: `100`. |

**Architecture**
```
Embedding → mean-pool (non-padding tokens) → Linear(emb_dim, hidden) → Dropout → ReLU → Linear(hidden, num_classes)
```

**Forward input/output**
- Input: `LongTensor (batch, seq_len)`
- Output: `FloatTensor (batch, num_classes)` — raw logits

---

### `LSTM(nn.Module)`

Bidirectional LSTM classifier. Encodes each sequence with a multi-layer BiLSTM and classifies using the concatenated final hidden states from both directions.

**Constructor parameters**
| Parameter | Type | Description |
|---|---|---|
| `weights_matrix` | `Tensor` | Pretrained embedding matrix `(vocab_size, emb_dim)`. |
| `num_classes` | `int` | Number of output classes. |
| `hidden_size` | `int` | Hidden units per direction. Default: `256`. |
| `num_layers` | `int` | Number of stacked LSTM layers. Default: `1`. |
| `dropout_prob` | `float` | Dropout between LSTM layers (only active when `num_layers > 1`) and before the final linear layer. |
| `padding_idx` | `int` | Index of the `<PAD>` token. Default: `0`. |

**Architecture**
```
Embedding → pack_padded_sequence → BiLSTM → concat(forward_hidden[-1], backward_hidden[-1]) → Dropout → Linear(hidden*2, num_classes)
```

Uses `pack_padded_sequence` with `enforce_sorted=False` so batches do not need to be sorted by length.

**Forward input/output**
- Input: `LongTensor (batch, seq_len)`
- Output: `FloatTensor (batch, num_classes)` — raw logits

## Dependencies
`torch`, `torch.nn`, `sklearn.feature_extraction.text`, `sklearn.linear_model`, `sklearn.pipeline`, `sklearn.metrics`, `torch.nn.utils.rnn`
