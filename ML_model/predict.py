import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import CreMLPHead

DNA_TO_INT = {b: i for i, b in enumerate("ACGTN")}
def encode_dna(seq):
    seq = str(seq).upper()[:34].ljust(34, 'N')
    return torch.tensor([DNA_TO_INT.get(base, 4) for base in seq], dtype=torch.long)

class TestDataset(Dataset):
    def __init__(self, csv_file, embedding_file):
        self.df = pd.read_csv(csv_file)
        self.embeddings = torch.load(embedding_file)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.embeddings[row['sequence']], encode_dna(row['lox_site']), torch.tensor(row['label'], dtype=torch.float32)

def run_test():
    device = torch.device("cpu")
    model = CreMLPHead().to(device)
    model.load_state_dict(torch.load('ML_model/cre_lox_model_8M_best.pth', map_location=device))
    model.eval()

    test_set = TestDataset('ML_model/test.csv', 'ML_model/esm2_8M_embeddings.pt')
    loader = DataLoader(test_set, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for p_vec, d_idx, label in loader:
            output = model(p_vec, d_idx).squeeze()
            if output.dim() == 0: output = output.unsqueeze(0)
            all_preds.extend(output.numpy())
            all_labels.extend(label.numpy())

    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    
    print("\n" + "="*35)
    print("8M MODEL: UNSEEN TEST SET RESULTS")
    print("="*35)
    print(f"Accuracy:  {accuracy_score(all_labels, binary_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, binary_preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, binary_preds):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(all_labels, all_preds):.4f}")

if __name__ == "__main__":
    run_test()