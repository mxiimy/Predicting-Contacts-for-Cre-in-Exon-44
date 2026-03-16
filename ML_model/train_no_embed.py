from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from model_no_embed import CreMLPNoEmbed, protein_to_tensor
except ImportError:
    from ML_model.model_no_embed import CreMLPNoEmbed, protein_to_tensor


DNA_TO_INT = {b: i for i, b in enumerate("ACGTN")}


def resolve_path(path_str):
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    cwd_candidate = Path.cwd() / candidate
    script_candidate = script_dir / candidate
    project_candidate = script_dir.parent / candidate

    for path in [cwd_candidate, script_candidate, project_candidate]:
        if path.exists():
            return path

    return candidate


def load_dna_lookup(dna_csv_path):
    dna_path = resolve_path(dna_csv_path)
    dna_df = pd.read_csv(dna_path)

    col_map = {str(c).strip().lower(): c for c in dna_df.columns}
    name_col = col_map.get("name")
    seq_col = col_map.get("sequence")

    if name_col is None or seq_col is None:
        dna_df = pd.read_csv(dna_path, header=None, names=["sequence", "name"])
        name_col = "name"
        seq_col = "sequence"

    return dict(zip(dna_df[name_col].astype(str).str.strip(), dna_df[seq_col].astype(str).str.strip()))


def dna_to_tensor(name, dna_lookup, fixed_len=34):
    seq = dna_lookup.get(str(name), "N" * fixed_len)
    seq = str(seq).upper()[:fixed_len].ljust(fixed_len, "N")
    return torch.tensor([DNA_TO_INT.get(b, 4) for b in seq], dtype=torch.long)


class NoEmbedCreDataset(Dataset):
    def __init__(self, csv_file, dna_csv_file="full_lox_sites.csv", max_protein_len=1024):
        self.df = pd.read_csv(resolve_path(csv_file))
        self.dna_lookup = load_dna_lookup(dna_csv_file)
        self.max_protein_len = max_protein_len

        col_map = {c.strip().lower(): c for c in self.df.columns}

        self.seq_col = None
        for key in ["protein sequence", "protein_sequence", "sequence"]:
            if key in col_map:
                self.seq_col = col_map[key]
                break

        self.dna_col = None
        for key in ["source_sheet", "lox_site", "lox site", "site"]:
            if key in col_map:
                self.dna_col = col_map[key]
                break

        self.label_col = None
        for key in ["label", "target"]:
            if key in col_map:
                self.label_col = col_map[key]
                break

        missing = []
        if self.seq_col is None:
            missing.append("sequence column")
        if self.dna_col is None:
            missing.append("DNA site column")
        if self.label_col is None:
            missing.append("label column")

        if missing:
            raise ValueError(f"Missing required columns in {csv_file}: {', '.join(missing)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prot_indices = protein_to_tensor(row[self.seq_col], max_len=self.max_protein_len)
        dna_indices = dna_to_tensor(row[self.dna_col], self.dna_lookup)
        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return prot_indices, dna_indices, label


def evaluate_model(model, loader, device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for p_idx, d_idx, label in loader:
            p_idx, d_idx, label = p_idx.to(device), d_idx.to(device), label.to(device)
            logits = model(p_idx, d_idx).squeeze(1)
            loss = criterion(logits, label)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds > 0.5).astype(int)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float("nan")

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, binary_preds),
        "precision": precision_score(all_labels, binary_preds, zero_division=0),
        "recall": recall_score(all_labels, binary_preds, zero_division=0),
        "f1": f1_score(all_labels, binary_preds, zero_division=0),
        "auc_roc": auc,
    }
    return metrics


def train_one_fold(
    train_loader,
    val_loader,
    device,
    max_protein_len=1024,
    epochs=50,
    learning_rate=3e-4,
    weight_decay=1e-4,
    early_stopping_patience=8,
    grad_clip_max_norm=1.0,
):
    model = CreMLPNoEmbed(protein_max_len=max_protein_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_metrics = None
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for p_idx, d_idx, label in train_loader:
            p_idx, d_idx, label = p_idx.to(device), d_idx.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(p_idx, d_idx).squeeze(1)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            train_loss += loss.item()

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics.copy()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, LR: {current_lr:.2e}"
            )

        if no_improve_epochs >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch+1} (no val loss improvement for {early_stopping_patience} epochs)")
            break

    return best_metrics, best_model_state


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_protein_len = 512
    batch_size = 32
    epochs = 50
    learning_rate = 3e-4
    weight_decay = 1e-4
    early_stopping_patience = 8
    grad_clip_max_norm = 1.0

    full_dataset = NoEmbedCreDataset("train.csv", dna_csv_file="full_lox_sites.csv", max_protein_len=max_protein_len)
    print(f"Total dataset size: {len(full_dataset)} samples")
    print(
        f"Config: max_protein_len={max_protein_len}, batch_size={batch_size}, epochs={epochs}, "
        f"lr={learning_rate}, weight_decay={weight_decay}"
    )

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

        train_subset = Subset(full_dataset, train_ids)
        val_subset = Subset(full_dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        metrics, model_state = train_one_fold(
            train_loader,
            val_loader,
            device,
            max_protein_len=max_protein_len,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            grad_clip_max_norm=grad_clip_max_norm,
        )
        fold_results.append(metrics)

        print(f"\nFold {fold + 1} Best Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

        torch.save(model_state, f"cre_model_no_embed_fold{fold+1}.pth")

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for metric in ["loss", "accuracy", "precision", "recall", "f1", "auc_roc"]:
        values = np.array([r[metric] for r in fold_results], dtype=np.float64)
        print(f"{metric.upper():12s}: {np.nanmean(values):.4f} ± {np.nanstd(values):.4f}")

    print(f"\n{'='*60}")
    print("Training final model on full dataset...")
    print(f"{'='*60}")

    final_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    final_model = CreMLPNoEmbed(protein_max_len=max_protein_len).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_final_loss = float("inf")
    best_final_state = None
    no_improve_epochs = 0

    for epoch in range(epochs):
        final_model.train()
        total_loss = 0

        for p_idx, d_idx, label in final_loader:
            p_idx, d_idx, label = p_idx.to(device), d_idx.to(device), label.to(device)

            optimizer.zero_grad()
            logits = final_model(p_idx, d_idx).squeeze(1)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(final_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_final_loss:
            best_final_loss = avg_loss
            best_final_state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping final training at epoch {epoch+1}")
            break

    if best_final_state is not None:
        final_model.load_state_dict(best_final_state)

    torch.save(final_model.state_dict(), "final_cre_model_no_embed.pth")
    print("\nFinal model saved to final_cre_model_no_embed.pth")


if __name__ == "__main__":
    train()
