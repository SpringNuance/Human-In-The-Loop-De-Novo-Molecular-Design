import torch
from torch import nn

class ScoreRegressionModel(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 1), # logit
            nn.Sigmoid() # probability of drd2
        )
    
    def forward(self, features):
        # Compute strength scores
        theta = self.fc(features) # Features is Morgan Fingerprints of smiles
        # theta is the predicted strength of smiles
        return theta

    def predict_proba(self, features):
        # Compute strength scores
        theta = self.fc(features) # Features is Morgan Fingerprints of smiles
        # theta is the predicted strength of smiles
        return theta