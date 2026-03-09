import torch
import pandas as pd
import esm

# load ESM-2 650M model and alphabet
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter() # Transform sequences into numerical format for ESM-2

DNA_TO_INT = {b: i for i, b in enumerate("ACGTN")} # Encode DNA
dna_df = pd.read_csv('full_lox_sites.csv')
DNA_LOOKUP = dict(zip(dna_df['name'], dna_df['sequence'])) # Map sequence names to strings

def get_esm_tokens(sequence_list):
    """Converts protein sequences into tokens compatible with ESM-2"""
    cleaned_seqs = [str(seq).replace('*', '').upper() for seq in sequence_list]
    data = [("protein", seq) for seq in cleaned_seqs]
    # batch_converter handles padding and start/stop tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    return batch_tokens
    
def dna_to_tensor(name):
    """Process the DNA so that each lox site is exactly 34 nucleotides long. If it is 
    too short, pad it with N which means any nucleotide. This shouldn't happen at all
    but acts as a failsafe
    """
    seq = DNA_LOOKUP.get(name, "N"*34)
    seq = str(seq).upper()[:34].ljust(34, 'N')
    return torch.tensor([DNA_TO_INT.get(b, 4) for b in seq], dtype=torch.long)