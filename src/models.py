import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence

class BaselineModel:
    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2)
            ),
            LogisticRegression(
                solver='liblinear',
                C=1.0
            )
        )

    def train(self, X_train, y_train):
        print("Training Baseline Model (TF-IDF + Logistic Regression)...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_dev, y_dev):
        y_pred = self.model.predict(X_dev)
        f1 = f1_score(y_dev, y_pred, average='macro')
        print(f'Baseline Macro F1 score: {f1:.4f}')
        return f1

class CBOW(nn.Module):
    def __init__(self, weights_matrix, num_classes, dropout_prob, padding_idx, hidden_size=100):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, padding_idx=padding_idx)
        self.embedding_dim = weights_matrix.shape[1]

        self.linear1 = nn.Linear(self.embedding_dim, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

        self.padding_idx = padding_idx

    def forward(self, x):
        non_pad_mask = (x != self.padding_idx)
        lengths = non_pad_mask.sum(dim=1).float().clamp(min=1).unsqueeze(1)

        embedded = self.embedding(x)
        x = torch.sum(embedded, dim=1) / lengths

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class LSTM(nn.Module):
    def __init__(self, weights_matrix, num_classes, hidden_size=256, num_layers=1, dropout_prob=0.5, padding_idx=0):
        super().__init__()

        weights_tensor = weights_matrix.clone().detach() # Avoid warning
        self.embedding = nn.Embedding.from_pretrained(
            weights_tensor,
            padding_idx=padding_idx
        )
        self.emb_dim = weights_matrix.shape[1]

        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.dropout = nn.Dropout(dropout_prob)
        self.padding_idx = padding_idx

    def forward(self, x):
        lengths = (x != self.padding_idx).sum(dim=1).cpu()

        embeds = self.embedding(x) # [Batch, Seq, Emb]
        
        # Pack padded sequence
        # Note: We need to sort by length for pack_padded_sequence if enforce_sorted=True (default). 
        # But here we set enforce_sorted=False.
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Handle bidirectional: concat forward and backward hidden states from the last layer
        cat_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        output = self.dropout(cat_hidden)
        output = self.fc(output)

        return output
