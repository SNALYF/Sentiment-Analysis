# src/utils.py

Utility helpers for reproducibility and device selection.

## Functions

### `set_seed(seed=572)`

Fixes the random state across all relevant libraries and CUDA to ensure reproducible results.

**Parameters**
- `seed` (`int`): The seed value. Default: `572`.

**What it sets**
| Target | Call |
|---|---|
| Python built-in `random` | `random.seed(seed)` |
| NumPy | `np.random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch CUDA | `torch.cuda.manual_seed(seed)` |
| cuDNN determinism | `torch.backends.cudnn.deterministic = True` |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = False` |
| Python hash seed | `os.environ['PYTHONHASHSEED'] = str(seed)` |

---

### `get_device()`

Detects and returns the best available compute device.

**Returns**
- `torch.device('cuda')` if a CUDA-capable GPU is available.
- `torch.device('cpu')` otherwise.

Prints the selected device to stdout.

## Dependencies
`torch`, `numpy`, `random`, `os`
