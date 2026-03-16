import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import CreMLPHead

DNA_TO_INT = {b: i for i, b in enumerate("ACGTN")}

def encode_dna(seq):
    """Converts lox site string to tensor of indices."""
    seq = str(seq).upper()[:34].ljust(34, 'N') # Ensure 34bp length
    return torch.tensor([DNA_TO_INT.get(base, 4) for base in seq], dtype=torch.long)

class EmbeddedCreDataset(Dataset):
    def __init__(self, csv_file, embedding_file):
        self.df = pd.read_csv(csv_file)
        self.embeddings = torch.load(embedding_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        prot_seq = row['sequence']
        prot_vec = self.embeddings[prot_seq]
        
        dna_indices = encode_dna(row['lox_site'])
        
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return prot_vec, dna_indices, label

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for p_vec, d_idx, label in loader:
            p_vec, d_idx, label = p_vec.to(device), d_idx.to(device), label.to(device)
            output = model(p_vec, d_idx).squeeze()
            
            if output.dim() == 0: output = output.unsqueeze(0)
            
            loss = criterion(output, label)
            total_loss += loss.item()
            
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, binary_preds),
        'precision': precision_score(all_labels, binary_preds, zero_division=0),
        'recall': recall_score(all_labels, binary_preds, zero_division=0),
        'f1': f1_score(all_labels, binary_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # File paths
    dataset_path = 'ML_model/train.csv'
    embedding_path = 'ML_model/simple_esm2_embeddings.pt'
    
    full_dataset = EmbeddedCreDataset(dataset_path, embedding_path)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n--- Fold {fold+1} ---")
        
        train_sub = torch.utils.data.Subset(full_dataset, train_idx)
        val_sub = torch.utils.data.Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

        model = CreMLPHead().to(device)
        
        # Optimizer and Scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        # Lowers LR by half every 10 epochs to help the model converge smoothly
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        criterion = nn.BCELoss()

        for epoch in range(30):
            model.train()
            for p_vec, d_idx, label in train_loader:
                p_vec, d_idx, label = p_vec.to(device), d_idx.to(device), label.to(device)
                
                optimizer.zero_grad()
                output = model(p_vec, d_idx).squeeze()
                
                if output.dim() == 0: output = output.unsqueeze(0)
                
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            
            # Step the learning rate scheduler
            scheduler.step()

        metrics = evaluate_model(model, val_loader, device)
        fold_results.append(metrics)
        print(f"Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.4f}")

    # Summary
    print(f"\n{'='*30}\nCV Results Summary\n{'='*30}")
    for m in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        vals = [r[m] for r in fold_results]
        print(f"{m.upper():10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

if __name__ == "__main__":
    train()