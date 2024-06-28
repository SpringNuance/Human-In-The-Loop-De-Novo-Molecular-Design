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
            nn.Linear(256, 1), # logit
            nn.Sigmoid() # probability of drd2
        )
    
    def forward(self, features_A, features_B, features_C):
        # Compute strength scores
        theta_A = self.fc(features_A) # Features A is Morgan Fingerprints of smiles 1 
        # theta_A is the predicted strength of smiles 1
        theta_B = self.fc(features_B) # Features B is Morgan Fingerprints of smiles 2
        # theta_B is the predicted strength of smiles 2
        theta_C = self.fc(features_C) # Features C is Morgan Fingerprints of smiles 3
        # theta_C is the predicted strength of smiles 3

        # Concatenate the scores into a tensor
        scores = torch.cat([theta_A, theta_B, theta_C], dim=1) # shape (batch_size, 3)
        
        # Apply softmax to get ranking probabilities
        ranking_scores = torch.softmax(scores, dim=1) # shape (batch_size, 3)

        return ranking_scores