# src/eda.py

Exploratory Data Analysis on the training set. Generates and saves two visualizations.

## Functions

### `perform_eda(df, output_dir='img')`

Analyzes the training DataFrame and saves two PNG plots to `output_dir`.

**Parameters**
- `df` (`DataFrame`): Training set with `content` and `rating` columns.
- `output_dir` (`str`): Directory where plots are saved. Created automatically if it does not exist. Default: `'img'`.

**Outputs**

| File | Description |
|---|---|
| `rating_distribution.png` | Bar chart of review counts per star rating (1star–5star), useful for checking class balance. |
| `review_length_distribution.png` | Stacked histogram with KDE of word counts per review, broken down by rating. |

**Side effects**
- Creates `output_dir` if it does not exist.
- Prints the save path for each plot to stdout.
- Does not modify the input DataFrame (operates on an internal copy).

## Dependencies
`matplotlib`, `seaborn`, `os`, `pandas`
