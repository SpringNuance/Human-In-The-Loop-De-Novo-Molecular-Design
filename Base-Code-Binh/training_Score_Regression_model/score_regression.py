import torch
from torch import nn

class ScoreRegressionModel(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 1)   # Output a single strength score per entity (lambda)
        )
    
    def forward(self, features):
        # Compute strength scores
        theta = self.fc(features) # Features is Morgan Fingerprints of smiles
        # theta is the predicted strength of smiles
        return torch.sigmoid(theta) # Sigmoid to ensure the output is between 0 and 1