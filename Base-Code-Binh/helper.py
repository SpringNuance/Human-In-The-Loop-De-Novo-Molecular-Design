import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import os
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import AllChem

from training_Score_Regression_model.score_regression import ScoreRegressionModel
from training_Bradley_Terry_model.bradley_terry import BradleyTerryModel
from training_Rank_ListNet_model.rank_listnet import RankListNetModel

from scripts.write_config_score_regression import write_REINVENT_config_score_regression
from scripts.write_config_bradley_terry import write_REINVENT_config_bradley_terry
from scripts.write_config_rank_listnet import write_REINVENT_config_rank_listnet

from itertools import combinations
from itertools import product

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def write_REINVENT_config(feedback_type, reinvent_dir, jobid, jobname, 
                          REINVENT_round_output_dir, conf_filename):
    """
        Write the REINVENT configuration file
    """

    if feedback_type == "scoring":
        configuration_JSON_path =\
            write_REINVENT_config_score_regression(reinvent_dir, jobid, jobname,
                                                REINVENT_round_output_dir, conf_filename)
    elif feedback_type == "comparing":
        configuration_JSON_path =\
            write_REINVENT_config_bradley_terry(reinvent_dir, jobid, jobname,
                                                REINVENT_round_output_dir, conf_filename)
    elif feedback_type == "ranking":
        configuration_JSON_path =\
            write_REINVENT_config_rank_listnet(reinvent_dir, jobid, jobname,
                                                REINVENT_round_output_dir, conf_filename)
    else:
        raise ValueError("Invalid model type")
    return configuration_JSON_path

        
def change_config_json(configuration, REINVENT_n_steps, batch_size, current_model_path):
    # write specified number of RL optimization steps in configuration
    # (example: if num_rounds = 5 (rounds) and Reinvent REINVENT_n_steps = 100, we will do 5*100 RL optimization steps)

    configuration["parameters"]["reinforcement_learning"]["n_steps"] = REINVENT_n_steps

    configuration["parameters"]["reinforcement_learning"]["batch_size"] = batch_size
    configuration["parameters"]["reinforcement_learning"]["sigma"] = batch_size

    # write the model path at current HITL round
    configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
    
    configuration_scoring_function[0]["specific_parameters"]["model_pretrained_path"] = os.path.abspath(current_model_path)

    configuration["parameters"]["scoring_function"]["parameters"] = configuration_scoring_function

    return configuration
    
def load_feedback_model(feedback_type, feedback_model_path):
    if feedback_type == "scoring":
        feedback_model = ScoreRegressionModel()
        # Load the state dict
        feedback_model.load_state_dict(torch.load(feedback_model_path))
        print(f"Loading Score Regression model from {feedback_model_path}")
    elif feedback_type == "comparing":
        feedback_model = BradleyTerryModel()
        # Load the state dict
        feedback_model.load_state_dict(torch.load(feedback_model_path))
        print(f"Loading Bradley Terry model successfully from {feedback_model_path}")
    elif feedback_type == "ranking":
        feedback_model = RankListNetModel()
        # Load the state dict
        feedback_model.load_state_dict(torch.load(feedback_model_path))
        print(f"Loading Rank ListNet model successfully from {feedback_model_path}")
    else:
        raise ValueError("Invalid model type")
    
    return feedback_model

def compute_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    else:
        return np.zeros((2048,), dtype=int)

def load_drd2_dataset(feedback_type, data_path):
    if feedback_type == "scoring":
        # Load the training data for Score Regression model
        dataframe = pd.read_csv(data_path)
        smiles_list = dataframe["smiles"].values
        features = np.array([compute_fingerprints(smiles) for smiles in smiles_list])
        label_proba = dataframe["label_proba"].to_numpy()
        label_binary = dataframe["label_binary"].to_numpy()
        outputs = {
            "smiles": smiles_list,
            "features": features,
            "label_proba": label_proba,
            "label_binary": label_binary
        }
        return outputs
    
    elif feedback_type == "comparing":
        # Load the training data for Bradley Terry model
        dataframe = pd.read_csv(data_path)
        smiles_1_list = dataframe["smiles_1"].values
        smiles_2_list = dataframe["smiles_2"].values
        features_1 = np.array([compute_fingerprints(smiles) for smiles in smiles_1_list])
        features_2 = np.array([compute_fingerprints(smiles) for smiles in smiles_2_list])
        label_1_proba = dataframe["label_1_proba"].values
        label_2_proba = dataframe["label_2_proba"].values
        label_1_binary = dataframe["label_1_binary"].values
        label_2_binary = dataframe["label_2_binary"].values
        compare_proba = dataframe["compare_proba"].values
        compare_binary = dataframe["compare_binary"].values

        outputs = {
            "smiles_1": smiles_1_list,
            "smiles_2": smiles_2_list,
            "features_1": features_1,
            "features_2": features_2,
            "label_1_proba": label_1_proba,
            "label_2_proba": label_2_proba,
            "label_1_binary": label_1_binary,
            "label_2_binary": label_2_binary,
            "compare_proba": compare_proba,
            "compare_binary": compare_binary
        }
        return outputs
    
    elif feedback_type == "ranking":
        # Load the training data for Rank ListNet model
        dataframe = pd.read_csv(data_path)
        smiles_1_list = dataframe["smiles_1"].values
        smiles_2_list = dataframe["smiles_2"].values
        smiles_3_list = dataframe["smiles_3"].values
        features_1 = np.array([compute_fingerprints(smiles) for smiles in smiles_1_list])
        features_2 = np.array([compute_fingerprints(smiles) for smiles in smiles_2_list])
        features_3 = np.array([compute_fingerprints(smiles) for smiles in smiles_3_list])
        label_1_proba = dataframe["label_1_proba"].values
        label_2_proba = dataframe["label_2_proba"].values
        label_3_proba = dataframe["label_3_proba"].values
        label_1_binary = dataframe["label_1_binary"].values
        label_2_binary = dataframe["label_2_binary"].values
        label_3_binary = dataframe["label_3_binary"].values
        label_1_softmax = dataframe["label_1_softmax"].values
        label_2_softmax = dataframe["label_2_softmax"].values
        label_3_softmax = dataframe["label_3_softmax"].values
        label_1_rank = dataframe["label_1_rank"].values
        label_2_rank = dataframe["label_2_rank"].values
        label_3_rank = dataframe["label_3_rank"].values
        outputs = {
            "smiles_1": smiles_1_list,
            "smiles_2": smiles_2_list,
            "smiles_3": smiles_3_list,
            "features_1": features_1,
            "features_2": features_2,
            "features_3": features_3,
            "label_1_proba": label_1_proba,
            "label_2_proba": label_2_proba,
            "label_3_proba": label_3_proba,
            "label_1_binary": label_1_binary,
            "label_2_binary": label_2_binary,
            "label_3_binary": label_3_binary,
            "label_1_softmax": label_1_softmax,
            "label_2_softmax": label_2_softmax,
            "label_3_softmax": label_3_softmax,
            "label_1_rank": label_1_rank,
            "label_2_rank": label_2_rank,
            "label_3_rank": label_3_rank
        }
        return outputs
    else:
        raise ValueError("Invalid model type")

def read_scaffold_result(scaffold_memory_path, choose_top_smiles):
    scaffold_df = pd.read_csv(scaffold_memory_path)

    data_len = len(scaffold_df)
    print("Number of SMILES in scaffold_memory.csv: ", data_len)
    colnames = scaffold_df.columns 
    scaffold = scaffold_df["Scaffold"].to_numpy()
    smiles = scaffold_df["SMILES"].to_numpy()
    human_component = scaffold_df["Human-Component"].to_numpy()
    raw_human_component = scaffold_df["raw_Human-Component"].to_numpy()
    total_score = scaffold_df["total_score"].to_numpy()

    # The scaffold memory is already sorted by total_score
    high_scoring_idx = np.arange(0, choose_top_smiles)

    # Only analyse highest scoring molecules
    smiles_high_score = smiles[high_scoring_idx]
    scaffold_high_score = scaffold[high_scoring_idx]
    human_component_high_score = human_component[high_scoring_idx]
    raw_human_component_high_score = raw_human_component[high_scoring_idx]
    total_score_high_score = total_score[high_scoring_idx]
    
    # print shape
    # print("Scaffold shape: ", scaffold_high_score.shape)
    # print("SMILES shape: ", smiles_high_score.shape)
    # print("Human component shape: ", human_component_high_score.shape)
    # print("Raw human component shape: ", raw_human_component_high_score.shape)
    # print("Total score shape: ", total_score_high_score.shape)
    
    scaffold_df = pd.DataFrame({
        "scaffold": scaffold_high_score,
        "smiles": smiles_high_score,
        "human_component": human_component_high_score,
        "raw_human_component": raw_human_component_high_score,
        "total_score": total_score_high_score
    })

    output_high_score = {
        "scaffold_df": scaffold_df,
        "scaffold": scaffold_high_score,
        "smiles": smiles_high_score,
        "human_component": human_component_high_score,
        "raw_human_component": raw_human_component_high_score,
        "total_score": total_score_high_score
    }
    return output_high_score

def human_score(drd2_oracle_model, smiles, sigma_noise):
    if smiles:
        if sigma_noise > 0:
            noise = np.random.normal(0, sigma_noise, 1).item()
            human_score = np.clip(drd2_oracle_model(smiles) + noise, 0, 1)
        else:
            human_score = drd2_oracle_model(smiles)
        return float(human_score)
    else:
        return 0.0
        
def smiles_human_score(smiles, sigma_noise=0.1):
    drd2_oracle_model = Oracle(name = 'DRD2')
    smiles_score_human = [human_score(drd2_oracle_model, smile, sigma_noise) for smile in smiles] 
    return smiles_score_human

def create_drd2_dataset(feedback_type, new_queried_smiles, 
                        new_queried_smiles_human_score,
                        new_queried_fps):
    
    if feedback_type == "scoring":
        smiles = new_queried_smiles
        label_proba = new_queried_smiles_human_score
        label_binary = np.array([1 if score > 0.5 else 0 for score in label_proba])
        outputs = {
            "smiles": smiles,
            "features": new_queried_fps,
            "label_proba": label_proba,
            "label_binary": label_binary
        }
        return outputs
    
    elif feedback_type == "comparing":
        
        num_new_queried_smiles, fps_dim = new_queried_fps.shape
        # Generate all repeated combinations of 2 out of len(new_queried_smiles)
        comb = list(product(range(num_new_queried_smiles), repeat=2))
        # We exclude the combinations where the indices are the same
        comb = [c for c in comb if c[0] != c[1]]

        C = len(comb) 

        smiles_1 = []
        smiles_2 = []
        
        features_1 = np.zeros((C, fps_dim))
        features_2 = np.zeros((C, fps_dim))

        label_1_proba = np.zeros(C)
        label_2_proba = np.zeros(C)
        label_1_binary = np.zeros(C)
        label_2_binary = np.zeros(C)

        compare_proba = np.zeros(C)
        compare_binary = np.zeros(C)

        for i, (idx1, idx2) in enumerate(comb):
            # We only want to compare different smiles
            if idx1 == idx2:
                continue
            smiles_1.append(new_queried_smiles[idx1])
            smiles_2.append(new_queried_smiles[idx2])
            features_1[i, :] = new_queried_fps[idx1, :]
            features_2[i, :] = new_queried_fps[idx2, :]
            label_1_proba[i] = new_queried_smiles_human_score[idx1]
            label_2_proba[i] = new_queried_smiles_human_score[idx2]

            label_1_binary[i] = 1 if label_1_proba[i] > 0.5 else 0
            label_2_binary[i] = 1 if label_2_proba[i] > 0.5 else 0

            compare_proba[i] =  sigmoid(label_1_proba[i] - label_2_proba[i])
            compare_binary[i] = 1 if compare_proba[i] > 0.5 else 0

        outputs = {
            "smiles_1": smiles_1,
            "smiles_2": smiles_2,
            "features_1": features_1,
            "features_2": features_2,
            "label_1_proba": label_1_proba,
            "label_2_proba": label_2_proba,
            "label_1_binary": label_1_binary,
            "label_2_binary": label_2_binary,
            "compare_proba": compare_proba,
            "compare_binary": compare_binary,
        }
        return outputs
    
    elif feedback_type == "ranking":
        num_new_queried_smiles, fps_dim = new_queried_fps.shape
        # Generate all unrepeated combinations of 3 out of len(new_queried_smiles)
        comb = list(combinations(range(num_new_queried_smiles), 3))
        C = len(comb)  # This is the number of combinations, which is binom(num_new_queried_smiles, 3)

        smiles_1 = []
        smiles_2 = []
        smiles_3 = []
        
        features_1 = np.zeros((C, fps_dim))
        features_2 = np.zeros((C, fps_dim))
        features_3 = np.zeros((C, fps_dim))

        label_1_proba = []
        label_2_proba = []
        label_3_proba = []

        label_1_binary = []
        label_2_binary = []
        label_3_binary = []

        label_1_softmax = []
        label_2_softmax = []
        label_3_softmax = []

        label_1_rank = []
        label_2_rank = []
        label_3_rank = []

        for i, (idx1, idx2, idx3) in enumerate(comb):
            smiles_1.append(new_queried_smiles[idx1])
            smiles_2.append(new_queried_smiles[idx2])
            smiles_3.append(new_queried_smiles[idx3])

            features_1[i, :] = new_queried_fps[idx1, :]
            features_2[i, :] = new_queried_fps[idx2, :]
            features_3[i, :] = new_queried_fps[idx3, :]

            label_1_proba.append(new_queried_smiles_human_score[idx1])
            label_2_proba.append(new_queried_smiles_human_score[idx2])
            label_3_proba.append(new_queried_smiles_human_score[idx3])

            label_1_binary.append(1 if label_1_proba[i] > 0.5 else 0)
            label_2_binary.append(1 if label_2_proba[i] > 0.5 else 0)
            label_3_binary.append(1 if label_3_proba[i] > 0.5 else 0)

            proba_list = [new_queried_smiles_human_score[idx1], 
                          new_queried_smiles_human_score[idx2], 
                          new_queried_smiles_human_score[idx3]]
            
            proba_softmax = nn.Softmax(dim=0)(torch.tensor(proba_list))

            # Now we need to rank them
            # Rank 1 has lowest value, Rank 3 has highest value
            ranks = np.argsort(np.argsort(proba_softmax)) + 1

            label_1_softmax.append(proba_softmax[0])
            label_2_softmax.append(proba_softmax[1])
            label_3_softmax.append(proba_softmax[2])

            label_1_rank.append(ranks[0])
            label_2_rank.append(ranks[1])
            label_3_rank.append(ranks[2])

        outputs = {
            "smiles_1": smiles_1,
            "smiles_2": smiles_2,
            "smiles_3": smiles_3,
            "features_1": features_1,
            "features_2": features_2,
            "features_3": features_3,
            "label_1_proba": label_1_proba,
            "label_2_proba": label_2_proba,
            "label_3_proba": label_3_proba,
            "label_1_binary": label_1_binary,
            "label_2_binary": label_2_binary,
            "label_3_binary": label_3_binary,
            "label_1_softmax": label_1_softmax,
            "label_2_softmax": label_2_softmax,
            "label_3_softmax": label_3_softmax,
            "label_1_rank": label_1_rank,
            "label_2_rank": label_2_rank,
            "label_3_rank": label_3_rank
        }
        return outputs
    else:
        raise ValueError("Invalid model type")
    
def combine_drd2_dataset(feedback_type, base_training_dataset_outputs, 
                        iteration_training_dataset_outputs):

    if feedback_type == "scoring":
        attributes = ["smiles", "features", "label_proba", "label_binary"]
        
        combined_outputs = {}
        for attribute in attributes:
            base_attribute = base_training_dataset_outputs[attribute]
            iteration_attribute = iteration_training_dataset_outputs[attribute]
            combined_outputs[attribute] = np.concatenate((base_attribute, iteration_attribute), axis=0)
        return combined_outputs
    
    elif feedback_type == "comparing":
        attributes = ["smiles_1", "smiles_2", "features_1", "features_2",
                      "label_1_proba", "label_2_proba", "label_1_binary", "label_2_binary",
                        "compare_proba", "compare_binary"]
        
        combined_outputs = {}
        for attribute in attributes:
            base_attribute = base_training_dataset_outputs[attribute]
            iteration_attribute = iteration_training_dataset_outputs[attribute]
            combined_outputs[attribute] = np.concatenate((base_attribute, iteration_attribute), axis=0)
        return combined_outputs
    
    elif feedback_type == "ranking":
        attributes = ["smiles_1", "smiles_2", "smiles_3", 
                      "features_1", "features_2", "features_3",
                      "label_1_proba", "label_2_proba", "label_3_proba",
                      "label_1_binary", "label_2_binary", "label_3_binary",
                      "label_1_softmax", "label_2_softmax", "label_3_softmax",
                      "label_1_rank", "label_2_rank", "label_3_rank"]
        
        combined_outputs = {}
        for attribute in attributes:
            base_attribute = base_training_dataset_outputs[attribute]
            iteration_attribute = iteration_training_dataset_outputs[attribute]
            combined_outputs[attribute] = np.concatenate((base_attribute, iteration_attribute), axis=0)
        return combined_outputs
    else:
        raise ValueError("Invalid model type")
    

def save_drd2_dataset(feedback_type, base_training_dataset_outputs, saving_path):

    if feedback_type == "scoring":
        dataframe = pd.DataFrame({
            "smiles": base_training_dataset_outputs["smiles"],
            "label_proba": base_training_dataset_outputs["label_proba"],
            "label_binary": base_training_dataset_outputs["label_binary"]
        })
        dataframe.to_csv(saving_path, index=False)

    elif feedback_type == "comparing":
        dataframe = pd.DataFrame({
            "smiles_1": base_training_dataset_outputs["smiles_1"],
            "smiles_2": base_training_dataset_outputs["smiles_2"],
            "label_1_proba": base_training_dataset_outputs["label_1_proba"],
            "label_2_proba": base_training_dataset_outputs["label_2_proba"],
            "label_1_binary": base_training_dataset_outputs["label_1_binary"],
            "label_2_binary": base_training_dataset_outputs["label_2_binary"],
            "compare_proba": base_training_dataset_outputs["compare_proba"],
            "compare_binary": base_training_dataset_outputs["compare_binary"]
        })
        dataframe.to_csv(saving_path, index=False)

    elif feedback_type == "ranking":
        dataframe = pd.DataFrame({
            "smiles_1": base_training_dataset_outputs["smiles_1"],
            "smiles_2": base_training_dataset_outputs["smiles_2"],
            "smiles_3": base_training_dataset_outputs["smiles_3"],
            "label_1_proba": base_training_dataset_outputs["label_1_proba"],
            "label_2_proba": base_training_dataset_outputs["label_2_proba"],
            "label_3_proba": base_training_dataset_outputs["label_3_proba"],
            "label_1_binary": base_training_dataset_outputs["label_1_binary"],
            "label_2_binary": base_training_dataset_outputs["label_2_binary"],
            "label_3_binary": base_training_dataset_outputs["label_3_binary"],
            "label_1_softmax": base_training_dataset_outputs["label_1_softmax"],
            "label_2_softmax": base_training_dataset_outputs["label_2_softmax"],
            "label_3_softmax": base_training_dataset_outputs["label_3_softmax"],
            "label_1_rank": base_training_dataset_outputs["label_1_rank"],
            "label_2_rank": base_training_dataset_outputs["label_2_rank"],
            "label_3_rank": base_training_dataset_outputs["label_3_rank"]
        })
        dataframe.to_csv(saving_path, index=False)
    else:
        raise ValueError("Invalid model type")


def retrain_feedback_model(feedback_type, feedback_model, training_outputs, epochs=1):
    """
        Retrain the model with the new data
    """
    if feedback_type == "scoring":
        feedback_model.train()
        features = training_outputs["features"]
        label_proba = training_outputs["label_proba"]
        optimizer = optim.Adam(feedback_model.parameters(), lr=0.0001)  
        criterion = nn.BCELoss()

        features_tensor = torch.tensor(features).float()  # Ensure dtype is float32 for features
        label_proba_tensor = torch.tensor(label_proba).float()  # Ensure dtype is float32 if regression, or long if classification

        train_dataset = TensorDataset(features_tensor, label_proba_tensor)

        batch_size = 16  # You can adjust the batch size as needed
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        feedback_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for features, labels_proba in train_loader:
                optimizer.zero_grad()
                output = feedback_model(features)
                loss = criterion(output, labels_proba.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(features)}')
        feedback_model.eval()
        return feedback_model
    
    elif feedback_type == "comparing":

        # When using Binary Cross-Entropy Loss (BCELoss) in neural networks, the input expected by the 
        # loss function is a list of probabilities, not binary values (0 or 1)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(feedback_model.parameters(), lr=0.0001)
        
        features_1 = training_outputs["features_1"]
        features_2 = training_outputs["features_2"]
        compare_binary = training_outputs["compare_binary"]

        features_1_tensor = torch.tensor(features_1).float()  
        features_2_tensor = torch.tensor(features_2).float()
        compare_binary_tensor = torch.tensor(compare_binary).float()

        train_dataset = TensorDataset(features_1_tensor, features_2_tensor, compare_binary_tensor)
       
        batch_size = 64  # You can adjust the batch size as needed
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        feedback_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for features_1, features_2, compare_binary in train_loader:
                optimizer.zero_grad()
                output = feedback_model(features_1, features_2)
                loss = criterion(output, compare_binary.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(features_1)}')

        feedback_model.eval()

        return feedback_model
    elif feedback_type == "ranking":

        # Correct Usage of KLDivLoss
        # Model Outputs: Should be log-probabilities.
        # True Labels: Should be probabilities.

        criterion = nn.KLDivLoss(reduction='batchmean')

        optimizer = optim.Adam(feedback_model.parameters(), lr=0.0001)

        features_1 = training_outputs["features_1"]
        features_2 = training_outputs["features_2"]
        features_3 = training_outputs["features_3"]
        
        label_1_softmax = training_outputs["label_1_softmax"]
        label_2_softmax = training_outputs["label_2_softmax"]
        label_3_softmax = training_outputs["label_3_softmax"]

        features_1_tensor = torch.tensor(features_1).float()  
        features_2_tensor = torch.tensor(features_2).float()
        features_3_tensor = torch.tensor(features_3).float()

        label_1_softmax_tensor = torch.tensor(label_1_softmax).float()
        label_2_softmax_tensor = torch.tensor(label_2_softmax).float()
        label_3_softmax_tensor = torch.tensor(label_3_softmax).float()

        train_dataset = TensorDataset(features_1_tensor, features_2_tensor, 
                                    features_3_tensor, label_1_softmax_tensor,
                                    label_2_softmax_tensor, label_3_softmax_tensor)

        batch_size = 64  
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        feedback_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for features_1, features_2, features_3, label_1_softmax, label_2_softmax, label_3_softmax in train_loader:

                optimizer.zero_grad()
                ranking_scores = feedback_model(features_1, features_2, features_3) # softmax scores
                true_label = torch.stack([label_1_softmax, label_2_softmax, label_3_softmax], dim=1)
                
                # Taking log 
                ranking_scores = torch.log(ranking_scores + 1e-12)

                loss = criterion(ranking_scores, true_label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(features_1)}')

        feedback_model.eval()

        return feedback_model
    else:
        raise ValueError("Invalid model type")

