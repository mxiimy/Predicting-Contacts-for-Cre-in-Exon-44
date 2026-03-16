import torch
import esm
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # 33 layers, 650 million parameters
batch_converter = alphabet.get_batch_converter()
# if gpu is available, use it otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# set to evaluation mode so it doesn't act like training
model.to(device).eval() 

def run_embedding_extraction(input_csvs, output_file):
    all_seqs = []
    script_dir = Path(__file__).resolve().parent

    for csv_path in input_csvs:
        path_candidate = Path(csv_path)
        if not path_candidate.is_absolute():
            cwd_path = Path.cwd() / path_candidate
            script_path = script_dir / path_candidate
            if cwd_path.exists():
                path_candidate = cwd_path
            elif script_path.exists():
                path_candidate = script_path

        if not path_candidate.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        df = pd.read_csv(path_candidate)

        # Support common sequence column names across datasets
        column_map = {col.strip().lower(): col for col in df.columns}
        seq_col = None
        for candidate in ["protein sequence", "protein_sequence", "sequence"]:
            if candidate in column_map:
                seq_col = column_map[candidate]
                break

        if seq_col is None:
            print(f"Skipping {path_candidate}: no sequence column found")
            continue

        all_seqs.extend(df[seq_col].dropna().astype(str).tolist())
    
    unique_seqs = list(set(all_seqs))
    embeddings_dict = {}
    
    print(f"Extracting 650M embeddings for {len(unique_seqs)} sequences on {device}")
    
    # Do not track gradients -> reduce memory usage
    with torch.no_grad():
        # Process one sequence at a time could be slow so maybe batch it? Depends on GPU memory limits
        for seq in tqdm(unique_seqs):
            # Remove stop codons (*) and ensure uppercase
            clean_seq = str(seq).replace('*', '').replace(' ', '').upper()
            if not clean_seq:
                continue
            
            data = [("protein", clean_seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            # Extract from the final layer (33)
            results = model(batch_tokens, repr_layers=[33])
            token_reps = results["representations"][33]
            
            # Mean pooling to get one 1280-dim vector per protein instead of one vector per AA
            # If low accuracy, change to one vector per AA -> much higher compute cost
            # Exclude start/stop tokens [1:-1]
            prot_vec = token_reps[0, 1:-1].mean(0).cpu()
            embeddings_dict[seq] = prot_vec

    torch.save(embeddings_dict, output_file)
    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    # Process both training and testing files at once
    files_to_process = ['train.csv', 'test.csv']
    run_embedding_extraction(files_to_process, 'esm2_650M_embeddings.pt')