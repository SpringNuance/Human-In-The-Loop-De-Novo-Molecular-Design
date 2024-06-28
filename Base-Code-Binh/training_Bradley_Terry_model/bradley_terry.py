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
            nn.Linear(256, 1), # logit
            nn.Sigmoid() # probability of drd2
        )
    
    # Clarification:
    # In bradley-terry model, we should compare the raw output (or logits)
    # to obtain the probability of the winning probability of the first item
    # However at the end of the Sequential, we have a nn.Sigmoid() layer, which tells that it is a probability
    # instead of a logit. However, the strength is the original score of the Oracle telling the probability
    # that smiles have drd2 or not, so we can treat the sigmoid layer as something to enforce the constraint
    # not to interpret it as a probability. 
    # Therefore, the output of sequential can be considered the strength of the smiles
    
    def forward(self, features_A, features_B):
        # Compute strength scores
        theta_A = self.fc(features_A) # Features A is Morgan Fingerprints of smiles 1
        # theta_A is the predicted strength of smiles 1
        theta_B = self.fc(features_B) # Features B is Morgan Fingerprints of smiles 2
        # theta_B is the predicted strength of smiles 2
        return torch.sigmoid(theta_A - theta_B)
    
    def predict_proba(self, features_A):
        # Compute strength scores
        theta_A = self.fc(features_A)
        return theta_A