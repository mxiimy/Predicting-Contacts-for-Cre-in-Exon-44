# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from model import CreMLPHead
# from train import EmbeddedCreDataset # Reuse the Dataset class from your train script

# def train_and_save():
#     # Detect GPU (HPC usually has CUDA)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Training final model on: {device}")

#     # 1. Load your 650M embeddings and full dataset
#     # Change these paths to your HPC file locations
#     dataset = EmbeddedCreDataset('train.csv', 'esm2_650M_embeddings.pt')
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # 2. Initialize Model and Optimizer
#     model = CreMLPHead().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#     criterion = nn.BCELoss()

#     # 3. Training Loop
#     epochs = 40 # Increased slightly for the larger model
#     model.train()
    
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for p_vec, d_idx, label in loader:
#             p_vec, d_idx, label = p_vec.to(device), d_idx.to(device), label.to(device)
            
#             optimizer.zero_grad()
#             output = model(p_vec, d_idx).squeeze()
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         scheduler.step()
#         if (epoch + 1) % 5 == 0:
#             print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f}")

#     # 4. Save the weights
#     save_path = "cre_lox_650M_final.pth"
#     torch.save(model.state_dict(), save_path)
#     print(f"Final model saved to {save_path}")

# if __name__ == "__main__":
#     train_and_save()

# SIMPLE EMBEDDINGS MODEL (for preliminary tuning on laptop):
import torch
import torch.nn as nn

class CreMLPHead(nn.Module):
    def __init__(self):
        super(CreMLPHead, self).__init__()
        
        # DNA Branch: 34 bases -> 8-dim embedding
        self.dna_emb = nn.Embedding(5, 8) 
        
        # Simple CNN -> keeps spatial info
        self.dna_cnn = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # 34 length * 16 channels = 544
        )
        
        # ESM-2 8M (320) + Flattened DNA (544) = 864
        combined_dim = 320 + 544
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, prot_vec, dna_indices):
        # DNA CNN
        dna_x = self.dna_emb(dna_indices).transpose(1, 2) 
        dna_feat = self.dna_cnn(dna_x)
        
        # Combine
        combined = torch.cat((prot_vec, dna_feat), dim=1)
        return self.mlp(combined)