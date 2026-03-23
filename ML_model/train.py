import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import CreMLPHead

# DNA Encoding
DNA_TO_INT = {b: i for i, b in enumerate("ACGTN")}

def encode_dna(seq):
    seq = str(seq).upper()[:34].ljust(34, 'N') 
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
    all_preds, all_labels = [], []
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

    # --- UPDATED PATHS FOR 8M RUN ---
    dataset_path = 'ML_model/train.csv'
    embedding_path = 'ML_model/esm2_8M_embeddings.pt' # Ensure this matches your 8M filename
    results_file = '8M_cross_validation_results.txt'
    
    full_dataset = EmbeddedCreDataset(dataset_path, embedding_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    with open(results_file, 'w') as f:
        f.write(f"CRE-LOX 8M TRAINING RUN: {datetime.datetime.now()}\n")
        f.write("="*40 + "\n")

        best_accuracy = 0.0

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
            print(f"\n--- Fold {fold+1} ---")
            
            train_sub = torch.utils.data.Subset(full_dataset, train_idx)
            val_sub = torch.utils.data.Subset(full_dataset, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

            # --- MODEL INIT ---
            # Ensure your model.py CreMLPHead accepts 320 for input_dim if 8M
            model = CreMLPHead().to(device) 
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
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
                scheduler.step()

            metrics = evaluate_model(model, val_loader, device)
            fold_results.append(metrics)
            
            log_str = (f"Fold {fold+1}: Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                       f"AUC: {metrics['auc_roc']:.4f}\n")
            f.write(log_str)
            print(f"Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.4f}")

            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                torch.save(model.state_dict(), 'cre_lox_model_8M_best.pth')

        f.write("\n" + "="*30 + "\n")
        f.write("8M CV RESULTS SUMMARY\n")
        f.write("="*30 + "\n")
        for m in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            vals = [r[m] for r in fold_results]
            summary_line = f"{m.upper():10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n"
            f.write(summary_line)
            print(summary_line.strip())

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    train()