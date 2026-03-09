import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocess import dna_to_tensor
from model import CreMLPHead

class EmbeddedCreDataset(Dataset):
    # Dataset pairs precomputed protein embeddings with DNA indices and labels.
    def __init__(self, csv_file, embedding_file):
        # CSV contains row-level supervision; PT file maps protein sequence -> ESM vector.
        self.df = pd.read_csv(csv_file)
        self.embeddings = torch.load(embedding_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Pull one sample, then build model-ready tensors.
        row = self.df.iloc[idx]
        prot_seq = row['Protein Sequence']
        prot_vec = self.embeddings[prot_seq]
        dna_indices = dna_to_tensor(row['source_sheet'])
        label = torch.tensor(row['Label'], dtype=torch.float32)
        return prot_vec, dna_indices, label

def evaluate_model(model, loader, device, criterion):
    """Evaluate model and return metrics"""
    model.eval()
    # Collect raw probabilities for thresholded metrics and AUC.
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for p_vec, d_idx, label in loader:
            p_vec, d_idx, label = p_vec.to(device), d_idx.to(device), label.to(device)
            output = model(p_vec, d_idx).squeeze()
            loss = criterion(output, label)
            total_loss += loss.item()
            
            # Store CPU values for sklearn metric computation.
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Fixed threshold for binary decision metrics.
    binary_preds = (all_preds > 0.5).astype(int)
    
    metrics = {
        # Average loss across validation mini-batches.
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, binary_preds),
        'precision': precision_score(all_labels, binary_preds, zero_division=0),
        'recall': recall_score(all_labels, binary_preds, zero_division=0),
        'f1': f1_score(all_labels, binary_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_preds)
    }
    return metrics

def train_one_fold(train_loader, val_loader, device, epochs=20):
    """Train model for one fold and return best validation metrics"""
    # Fresh model per fold avoids information leakage between folds.
    model = CreMLPHead().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    best_metrics = None
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for p_vec, d_idx, label in train_loader:
            p_vec, d_idx, label = p_vec.to(device), d_idx.to(device), label.to(device)
            
            # Standard forward/backward/update step.
            optimizer.zero_grad()
            output = model(p_vec, d_idx).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        
        # Track best model based on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics = val_metrics.copy()
            # Save an in-memory snapshot for this fold's best checkpoint.
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    return best_metrics, best_model_state

def train():
    # Train on GPU when available; otherwise fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load full dataset
    full_dataset = EmbeddedCreDataset('train_balanced.csv', 'esm2_650M_embeddings.pt')
    print(f"Total dataset size: {len(full_dataset)} samples")
    
    # K-Fold Cross Validation
    k_folds = 5
    # Shuffle keeps fold composition unbiased; seed makes runs reproducible.
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}")
        
        # Create data loaders for this fold
        train_subset = Subset(full_dataset, train_ids)
        val_subset = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        # No shuffle for validation so evaluation is deterministic.
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Train and evaluate
        metrics, model_state = train_one_fold(train_loader, val_loader, device, epochs=20)
        fold_results.append(metrics)
        
        print(f"\nFold {fold + 1} Best Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Save best model from this fold
        torch.save(model_state, f"cre_model_fold{fold+1}.pth")
    
    # Print average metrics across all folds
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        values = [r[metric] for r in fold_results]
        print(f"{metric.upper():12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Train final model on all data
    print(f"\n{'='*60}")
    print("Training final model on full dataset...")
    print(f"{'='*60}")
    final_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    final_model = CreMLPHead().to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Full-data training uses the same objective after CV selection.
    for epoch in range(20):
        final_model.train()
        total_loss = 0
        for p_vec, d_idx, label in final_loader:
            p_vec, d_idx, label = p_vec.to(device), d_idx.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = final_model(p_vec, d_idx).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20 - Loss: {total_loss/len(final_loader):.4f}")
    
    torch.save(final_model.state_dict(), "final_cre_model.pth")
    print("\nFinal model saved to final_cre_model.pth")

if __name__ == "__main__":
    train()