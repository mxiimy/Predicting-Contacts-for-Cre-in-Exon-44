import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights


AA_PAD_IDX = 0
AA_UNK_IDX = 1
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i + 2 for i, aa in enumerate(AA_VOCAB)}


def protein_to_tensor(sequence, max_len=1024):
    clean_seq = str(sequence).replace("*", "").replace(" ", "").upper()
    indices = [AA_TO_INT.get(aa, AA_UNK_IDX) for aa in clean_seq[:max_len]]

    if len(indices) < max_len:
        indices.extend([AA_PAD_IDX] * (max_len - len(indices)))

    return torch.tensor(indices, dtype=torch.long)


class ProteinSequenceEncoder(nn.Module):
    def __init__(self, emb_dim=64, out_dim=256):
        super(ProteinSequenceEncoder, self).__init__()
        vocab_size = len(AA_TO_INT) + 2  # PAD + UNK + amino acids
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=AA_PAD_IDX)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

    def forward(self, prot_indices):
        # prot_indices: [B, L]
        emb = self.embedding(prot_indices)  # [B, L, D]
        mask = prot_indices.ne(AA_PAD_IDX).unsqueeze(-1).float()  # [B, L, 1]

        summed = (emb * mask).sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        mean_pooled = summed / lengths

        return self.proj(mean_pooled)


class CreMLPNoEmbed(nn.Module):
    def __init__(self, protein_max_len=1024, protein_emb_dim=64, shared_dim=256):
        super(CreMLPNoEmbed, self).__init__()

        self.protein_max_len = protein_max_len
        
        # 1. Protein Encoder
        self.protein_encoder = ProteinSequenceEncoder(
            emb_dim=protein_emb_dim,
            out_dim=shared_dim # Output is 256
        )

        # 2. DNA Encoder (Projected to the same size as the protein)
        self.dna_emb = nn.Embedding(5, 4)  
        self.dna_proj = nn.Sequential(
            nn.Linear(34 * 4, 128),
            nn.SiLU(),
            nn.Linear(128, shared_dim) # Output is 256
        )

        # 3. We no longer concatenate. We multiply them (element-wise) to force interaction
        # combined_dim is now just shared_dim (256)
        
        self.attention = AttentionLayer(shared_dim)

        self.mlp = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, prot_indices, dna_indices):
        # Get 256-dim protein vector
        prot_vec = self.protein_encoder(prot_indices)
        
        # Get 256-dim DNA vector
        dna_flat = self.dna_emb(dna_indices).view(dna_indices.size(0), -1)
        dna_vec = self.dna_proj(dna_flat)

        # FORCE INTERACTION: Element-wise multiplication
        # If the protein features don't align with the DNA features, the values zero out.
        interacted = prot_vec * dna_vec 
        
        # Apply your attention layer to the interacted features
        attended = self.attention(interacted)

        return self.mlp(attended)
