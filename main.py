import os
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import src.utils as utils
import src.data_loader as data_loader
import src.eda as eda
import src.preprocessing as preprocessing
import src.models as models
import src.train as train

def main():
    # 1. Setup
    print("=== Setting up ===")
    utils.set_seed(572)
    device = utils.get_device()
    
    # 2. Data Loading
    print("\n=== Data Loading ===")
    data_loader.download_data('data')
    train_set, dev_set, test_set = data_loader.load_data('data/yelp_review')
    
    # 3. EDA
    print("\n=== EDA ===")
    eda.perform_eda(train_set, 'img')
    
    # 4. Preprocessing
    print("\n=== Preprocessing ===")
    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    # Fit on train labels
    y_train = label_encoder.fit_transform(train_set['rating'])
    y_dev = label_encoder.transform(dev_set['rating'])
    # Assign dummy labels for test set as we don't have them or they are hidden, logic from notebook
    test_set['rating'] = '1star' 
    y_test = label_encoder.transform(test_set['rating'])
    
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {label_encoder.classes_}")

    # Build Vocabulary
    print("Building vocabulary...")
    word2i = preprocessing.build_word2i(train_set['content'])
    vocab_size = len(word2i)
    print(f"Vocab size: {vocab_size}")
    
    # Build Embedding Matrix
    print("Building embedding matrix...")
    embedding_weights = preprocessing.build_embedding_matrix(word2i)
    
    # Create DataLoaders
    print("Creating DataLoaders...")
    batch_size = 64
    train_loader = preprocessing.create_data_loader(train_set, y_train, word2i, batch_size=batch_size, shuffle=True, device=device)
    dev_loader = preprocessing.create_data_loader(dev_set, y_dev, word2i, batch_size=batch_size, shuffle=False, device=device)
    
    # 5. Baseline Model
    print("\n=== Baseline Model ===")
    baseline = models.BaselineModel()
    baseline.train(train_set['content'], train_set['rating']) # Pass raw text and labels
    baseline.evaluate(dev_set['content'], dev_set['rating'])
    
    # 6. CBOW Model
    print("\n=== CBOW Model ===")
    cbow_model = models.CBOW(
        weights_matrix=embedding_weights,
        num_classes=num_classes,
        dropout_prob=0.5,
        padding_idx=word2i['<PAD>']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cbow_model.parameters(), lr=0.01)
    
    train.train_model(
        cbow_model, 
        train_loader, 
        dev_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=3, # keeping it short as per notebook
        model_path='model/best_cbow.pth'
    )
    
    # 7. BiLSTM Model
    print("\n=== BiLSTM Model ===")
    # Hyperparameters from "model_tuned" in notebook
    lstm_model = models.LSTM(
        weights_matrix=embedding_weights,
        num_classes=num_classes,
        hidden_size=256,
        num_layers=3,
        dropout_prob=0.5,
        padding_idx=word2i['<PAD>']
    )
    
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    
    train.train_model(
        lstm_model,
        train_loader,
        dev_loader,
        criterion,
        optimizer,
        device,
        num_epochs=10, # Notebook has 100 but with early stopping, 10 is enough for demo
        patience=3,
        model_path='model/best_bilstm.pth'
    )
    
    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
