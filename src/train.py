import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import copy
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, patience=3, model_path='model/best_model.pth'):
    """
    Generic training loop with early stopping.
    """
    model.to(device)
    train_loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    trigger_times = 0
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1 (Macro): {val_f1:.4f}')
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path} (F1: {val_f1:.4f})")
        else:
            trigger_times += 1
            print(f"Early stopping trigger: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
                
    print("Training finished.")
    # Load best model
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, loader, device):
    """
    Evaluates the model and returns predictions.
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    return f1, acc
