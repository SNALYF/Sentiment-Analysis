# Sentiment Analysis Pipeline

This project converts a research notebook into a modular machine learning pipeline for sentiment analysis on Yelp reviews. It includes data loading, exploratory data analysis (EDA), preprocessing, and training of multiple models (Baseline, CBOW, BiLSTM).

## Project Structure

```
.
├── main.py
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── img/
│   ├── rating_distribution.png
│   └── review_length_distribution.png
├── model/
│   ├── best_cbow.pth
│   └── best_bilstm.pth
└── data/
    └── yelp_review/
        ├── train.tsv
        ├── val.tsv
        └── test.tsv
```

## Pipeline Overview

The pipeline is orchestrated by `main.py` and consists of the following sequential steps:

1.  **Setup**: Initializes random seeds for reproducibility and selects the computing device (CPU or GPU).
2.  **Data Loading**: Downloads the Yelp review dataset from Google Drive (if not present) and loads the training, validation, and test sets into Pandas DataFrames.
3.  **EDA (Exploratory Data Analysis)**: Analyzes the training data and generates two plots in the `img/` directory:
    - `rating_distribution.png`: Shows the balance of star ratings.
    - `review_length_distribution.png`: Shows the distribution of review lengths.
4.  **Preprocessing**:
    - Encodes target labels (star ratings) into integers.
    - Builds a vocabulary from the training text.
    - Creates an embedding matrix using pre-trained Spacy vectors (`en_core_web_sm`).
    - Converts text data into PyTorch DataLoaders with padding.
5.  **Baseline Model**: Trains a TF-IDF Vectorizer + Logistic Regression model using scikit-learn to establish a baseline performance metric.
6.  **CBOW Model (Continuous Bag of Words)**: Trains a neural network that averages word embeddings and passes them through fully connected layers. The best model based on validation loss is saved to `model/best_cbow.pth`.
7.  **BiLSTM Model (Bidirectional LSTM)**: Trains a Bidirectional LSTM network to capture sequential dependencies in the text. It uses the same pre-trained embeddings and supports early stopping. The best model is saved to `model/best_bilstm.pth`.

## Usage

Run the entire pipeline using:

```bash
python3 main.py
```

Ensure all dependencies listed in `requirements.txt` are installed before running.