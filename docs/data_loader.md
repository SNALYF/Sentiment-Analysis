# src/data_loader.py

Handles downloading and loading the Yelp review dataset.

## Functions

### `download_data(output_dir='data')`

Downloads the Yelp review dataset from Google Drive into `output_dir/yelp_review/`. Skips the download if the target directory already exists.

Uses `gdown.download_folder` to fetch the entire folder from Google Drive.

**Parameters**
- `output_dir` (`str`): Root directory where the dataset folder will be placed. Default: `'data'`.

**Side effects**
- Creates `output_dir` if it does not exist.
- Prints download status and any errors to stdout.

---

### `load_data(data_dir='data/yelp_review')`

Reads `train.tsv`, `val.tsv`, and `test.tsv` from `data_dir` into Pandas DataFrames.

**Parameters**
- `data_dir` (`str`): Path to the directory containing the TSV files. Default: `'data/yelp_review'`.

**Returns**
- `train_set` (`DataFrame`): Training split.
- `dev_set` (`DataFrame`): Validation split.
- `test_set` (`DataFrame`): Test split.

Each DataFrame has two columns:
| Column | Type | Description |
|---|---|---|
| `content` | str | Raw review text |
| `rating` | str | Star label (`1star`–`5star`) |

**Raises**
- `FileNotFoundError`: If any of the TSV files are missing.

## Dependencies
`os`, `pandas`, `gdown`
