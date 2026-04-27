# src/preprocessing.py

Text preprocessing utilities: vocabulary construction, pretrained embedding matrix, and PyTorch DataLoader creation.

Loads `en_core_web_sm` from spaCy on import (downloads it automatically if missing).

## Functions

### `build_word2i(contents)`

Builds a word-to-integer index mapping from a collection of raw text strings.

**Parameters**
- `contents` (iterable of `str`): Review texts, typically `df['content']`.

**Returns**
- `word2i` (`dict[str, int]`): Mapping from lowercased token to integer index.

**Reserved indices**
| Token | Index |
|---|---|
| `<PAD>` | 0 |
| `<UNK>` | 1 |

All other words are assigned indices starting from 2, in the order they appear in the `Counter`.

---

### `build_embedding_matrix(word2i, emb_dim=300)`

Constructs a weight matrix by looking up spaCy's pretrained word vectors for every token in `word2i`.

**Parameters**
- `word2i` (`dict[str, int]`): Vocabulary mapping produced by `build_word2i`.
- `emb_dim` (`int`): Embedding dimension. Must match the spaCy model's vector size. Default: `300`.

**Returns**
- `torch.Tensor` of shape `(vocab_size, emb_dim)`, dtype `float32`.

**Behavior**
- Words found in spaCy's vocabulary use their pretrained vector.
- Words not found are initialized with random normal values (mean=0, std=0.6).
- The `<PAD>` row is always a zero vector.
- Prints the number of words with found embeddings.

---

### `create_data_loader(df, y, w2i, batch_size=32, shuffle=True, device='cpu')`

Converts a DataFrame and label array into a PyTorch `DataLoader` with padded sequences.

**Parameters**
- `df` (`DataFrame`): Must contain a `content` column.
- `y` (`array-like`): Integer-encoded labels aligned with `df`.
- `w2i` (`dict[str, int]`): Vocabulary mapping.
- `batch_size` (`int`): Number of samples per batch. Default: `32`.
- `shuffle` (`bool`): Whether to shuffle the dataset each epoch. Default: `True`.
- `device` (`str`): Unused at DataLoader creation time; tensors are moved to device during training.

**Returns**
- `DataLoader` yielding `(padded_inputs, labels)` tuples.
  - `padded_inputs`: `LongTensor` of shape `(batch, max_seq_len)`, padded with `<PAD>` index.
  - `labels`: `LongTensor` of shape `(batch,)`.

## Dependencies
`torch`, `numpy`, `collections.Counter`, `spacy`, `torch.utils.data`, `torch.nn.utils.rnn`, `subprocess`, `sys`
