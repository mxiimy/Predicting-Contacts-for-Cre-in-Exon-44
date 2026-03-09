import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """Feature attention mechanism to weight important interactions by compressing 
    features to 1/4 dimension then reexpanding.
    """
    def __init__(self, dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Learn attention weights for each feature
        attn_weights = self.attention(x)
        return x * attn_weights

class CreMLPHead(nn.Module):
    def __init__(self):
        super(CreMLPHead, self).__init__()
        
        # DNA Embedding 
        self.dna_emb = nn.Embedding(5, 4) # 34 bases * 4 nucleotides = 136
        
        # Combined dimension: ESM-2 650M (1280) + DNA (136)
        combined_dim = 1280 + 136
        
        # Attention mechanism to learn important feature interactions
        self.attention = AttentionLayer(combined_dim)
        
        # Deeper MLP with SiLU activation for better non-linearity
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 768),
            nn.BatchNorm1d(768), # Batch normalization to stabilize training
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.SiLU(),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, prot_vec, dna_indices):
        # DNA Feature Extraction
        dna_vec = self.dna_emb(dna_indices).view(dna_indices.size(0), -1)
        
        # Combine pre-computed protein vector with DNA vector
        combined = torch.cat((prot_vec, dna_vec), dim=1)
        
        # Apply attention to weight important feature interactions
        attended = self.attention(combined)
        
        return self.mlp(attended)