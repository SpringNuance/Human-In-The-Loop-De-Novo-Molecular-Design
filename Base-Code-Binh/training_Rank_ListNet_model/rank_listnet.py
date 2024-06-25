import torch
from torch import nn

class RankListNetModel(nn.Module):
    def __init__(self, feature_dim=2048):
        super(RankListNetModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 1)   # Output a single strength score per entity (lambda)
        )
    
    def forward(self, features_A, features_B, features_C):
        # Compute strength scores
        theta_A = self.fc(features_A) # Features A is Morgan Fingerprints of smiles 1
        # theta_A is the predicted strength of smiles 1
        theta_B = self.fc(features_B) # Features B is Morgan Fingerprints of smiles 2
        # theta_B is the predicted strength of smiles 2
        theta_C = self.fc(features_C) # Features C is Morgan Fingerprints of smiles 3
        # theta_C is the predicted strength of smiles 3
        ranking_scores = torch.softmax(torch.stack([theta_A, theta_B, theta_C]), dim=0)
        return ranking_scores