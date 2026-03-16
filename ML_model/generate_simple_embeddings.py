import torch
import esm
import pandas as pd
from tqdm import tqdm

# 1. Load the smallest ESM-2 model (8 million parameters)
# This model is fast and uses 320-dimensional embeddings
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def run_simple_embedding(input_files, output_file):
    # Collect all unique sequences from both files
    all_seqs = set()
    for f in input_files:
        try:
            df = pd.read_csv(f)
            # Find the sequence column (case-insensitive)
            col = [c for c in df.columns if 'sequence' in c.lower()][0]
            all_seqs.update(df[col].dropna().unique())
        except Exception as e:
            print(f"Could not process {f}: {e}")

    embeddings_dict = {}
    print(f"Extracting 8M embeddings for {len(all_seqs)} sequences on {device}...")

    with torch.no_grad():
        for seq in tqdm(all_seqs):
            clean_seq = str(seq).replace('*', '').upper()
            
            # Prepare data and move to device
            data = [("protein", clean_seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            # Extract from the last layer (layer 6 for the 8M model)
            results = model(batch_tokens, repr_layers=[6])
            token_reps = results["representations"][6]

            # Mean pooling: Average the amino acid vectors into one protein vector
            # [0, 1:-1] ignores the start/stop tokens
            mean_vec = token_reps[0, 1:-1].mean(0).cpu()
            embeddings_dict[seq] = mean_vec

    torch.save(embeddings_dict, output_file)
    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    run_simple_embedding(['ML_model/train.csv', 'ML_model/test.csv'], 'ML_model/esm2_embeddings.pt')