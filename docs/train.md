# src/train.py

Generic training and evaluation loops for PyTorch models.

## Functions

### `train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, patience=3, model_path='model/best_model.pth')`

Trains a model with early stopping based on validation loss. Saves the best checkpoint to disk.

**Parameters**
| Parameter | Type | Description |
|---|---|---|
| `model` | `nn.Module` | The model to train. |
| `train_loader` | `DataLoader` | Training batches. |
| `val_loader` | `DataLoader` | Validation batches used for early stopping and checkpointing. |
| `criterion` | loss function | e.g. `nn.CrossEntropyLoss()`. |
| `optimizer` | optimizer | e.g. `torch.optim.Adam(...)`. |
| `device` | `torch.device` | Target device for tensors and model. |
| `num_epochs` | `int` | Maximum number of training epochs. Default: `10`. |
| `patience` | `int` | Number of consecutive epochs without improvement before stopping. Default: `3`. |
| `model_path` | `str` | File path to save the best model weights. Parent directories are created automatically. Default: `'model/best_model.pth'`. |

**Returns**
- `model` (`nn.Module`): The model loaded with the best saved weights.

**Per-epoch output**
```
Epoch [n/N], Train Loss: X.XXXX, Val Loss: X.XXXX, Val F1 (Macro): X.XXXX
```

**Early stopping logic**
- Resets a counter when validation Macro F1 improves; increments otherwise.
- Stops training when the counter reaches `patience`.
- Always loads the best checkpoint before returning.

---

### `evaluate_model(model, loader, device)`

Runs inference on a DataLoader and returns performance metrics.

**Parameters**
| Parameter | Type | Description |
|---|---|---|
| `model` | `nn.Module` | Trained model. |
| `loader` | `DataLoader` | Dataset to evaluate on. |
| `device` | `torch.device` | Target device. |

**Returns**
- `f1` (`float`): Macro F1 score.
- `acc` (`float`): Accuracy.

Runs entirely under `torch.no_grad()`.

## Dependencies
`torch`, `torch.nn`, `sklearn.metrics`, `copy`, `os`
