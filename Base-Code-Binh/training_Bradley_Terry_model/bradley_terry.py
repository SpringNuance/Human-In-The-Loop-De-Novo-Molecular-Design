import torch
from torch import nn

class BradleyTerryModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 128),  
            nn.ReLU(),
            nn.Linear(128, 1)   # Output a single strength score per entity (lambda)
        )
    
    def forward(self, features_A, features_B):
        # Compute strength scores
        theta_A = self.fc(features_A) # Features A is Morgan Fingerprints of smiles 1
        # theta_A is the predicted strength of smiles 1
        theta_B = self.fc(features_B) # Features B is Morgan Fingerprints of smiles 2
        # theta_B is the predicted strength of smiles 2
        
        # natural base Bradley Terry formula
        # Pr(A better than B) = 1/(1 + exp(theta_B - theta_A)) 
        # The sigmoid function is
        # sigmoid(x) = 1/(1 + exp(-x))
        # => x = theta_A - theta_B
        # Compute the probability using the Bradley-Terry formula
        return torch.sigmoid(theta_A - theta_B)