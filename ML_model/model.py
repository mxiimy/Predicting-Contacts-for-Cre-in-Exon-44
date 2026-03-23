import torch
import torch.nn as nn

class CreMLPHead(nn.Module):
    def __init__(self):
        super(CreMLPHead, self).__init__()
        
        # DNA Branch: 34 bases -> 8-dim embedding
        self.dna_emb = nn.Embedding(5, 8) 
        
        # DNA CNN: processes the lox site
        self.dna_cnn = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # 34 length * 16 channels = 544
        )
        
        # ESM-2 8M (320) + DNA CNN (544) = 864
        combined_dim = 320 + 544 
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2), # Dropped from 0.4; 8M needs less regularization
            
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, prot_vec, dna_indices):
        # dna_indices: [batch, 34] -> [batch, 34, 8] -> [batch, 8, 34]
        dna_x = self.dna_emb(dna_indices).transpose(1, 2) 
        dna_feat = self.dna_cnn(dna_x)
        
        combined = torch.cat((prot_vec, dna_feat), dim=1)
        
        return self.mlp(combined)