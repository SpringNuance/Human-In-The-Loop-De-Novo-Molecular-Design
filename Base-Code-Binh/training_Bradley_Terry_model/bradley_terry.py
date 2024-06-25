import torch
from torch import nn

class BradleyTerryModel(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 1)   # Output a single strength score per entity (lambda)
        )
    
    def forward(self, features_A, features_B):
        # Compute strength scores
        theta_A = self.fc(features_A) # Features A is Morgan Fingerprints of smiles 1
        # theta_A is the predicted strength of smiles 1
        theta_B = self.fc(features_B) # Features B is Morgan Fingerprints of smiles 2
        return torch.sigmoid(theta_A - theta_B)