import torch
import numpy as np
from collections import Counter
import spacy
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import subprocess
import sys

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for the spaCy POS tagger\n"
          "(don't worry, this will only happen once)")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def build_word2i(contents):
    """
    Builds a word-to-index mapping from the list of contents.
    """
    counter = Counter()
    for content in contents:
        tokens = str(content).lower().split()
        counter.update(tokens)

    word2i = {}
    word2i['<PAD>'] = 0
    word2i['<UNK>'] = 1

    idx = 2
    for word, count in counter.items():
        word2i[word] = idx
        idx += 1

    return word2i

def build_embedding_matrix(word2i, emb_dim=300):
    """
    Builds the embedding matrix using Spacy's pretrained vectors.
    """
    vocab_size = len(word2i)
    weights_matrix = np.random.normal(scale=0.6, size=(vocab_size, emb_dim))
    weights_matrix[word2i['<PAD>']] = np.zeros((emb_dim,))
    
    found_count = 0
    for word, i in word2i.items():
        if word in nlp.vocab and nlp.vocab[word].has_vector:
            weights_matrix[i] = nlp.vocab[word].vector
            found_count += 1
            
    print(f"Found embeddings for {found_count} / {vocab_size} words.")
    return torch.tensor(weights_matrix, dtype=torch.float32)

def create_data_loader(df, y, w2i, batch_size=32, shuffle=True, device='cpu'):
    """
    Creates a PyTorch DataLoader.
    """
    pad_token_id = w2i.get('<PAD>', 0)
    unk_token_id = w2i.get('<UNK>', 1)

    def text_to_indices(text):
        tokens = str(text).lower().split()
        return [w2i.get(token, unk_token_id) for token in tokens]

    indices_list = [torch.tensor(text_to_indices(text)) for text in df['content']]
    padded_inputs = pad_sequence(indices_list, batch_first=True, padding_value=pad_token_id)

    labels = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(padded_inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
