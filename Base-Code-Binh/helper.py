import numpy as np
import pandas as pd
import torch
import os
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import AllChem

from training_Bradley_Terry_model.bradley_terry import BradleyTerryModel
from training_Rank_ListNet_model.rank_listnet import RankListNetModel
from training_Score_Regression_model.score_regression import ScoreRegressionModel

from scripts.write_config_bradley_terry import write_REINVENT_config_bradley_terry
from scripts.write_config_rank_listnet import write_REINVENT_config_rank_listnet
from scripts.write_config_score_regression import write_REINVENT_config_score_regression

def write_REINVENT_config(feedback_type, reinvent_dir, jobid, jobname, 
                          REINVENT_round_output_dir, conf_filename):
    """
        Write the REINVENT configuration file
    """

    if feedback_type == "scoring":
        configuration_JSON_path =\
            write_REINVENT_config_score_regression(reinvent_dir, jobid, jobname,
                                                REINVENT_round_output_dir, conf_filename, feedback_type)
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

        
def change_config_json(configuration, REINVENT_n_steps, current_model_path):
    # write specified number of RL optimization steps in configuration
    # (example: if num_rounds = 5 (rounds) and Reinvent REINVENT_n_steps = 100, we will do 5*100 RL optimization steps)

    configuration["parameters"]["reinforcement_learning"]["n_steps"] = REINVENT_n_steps
    # write the model path at current HITL round
    configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
    
    configuration_scoring_function[0]["specific_parameters"]["model_pretrained_path"] = current_model_path

    configuration["parameters"]["scoring_function"]["parameters"] = configuration_scoring_function

    return configuration
    
def load_feedback_model(feedback_type, feedback_model_path):
    if feedback_type == "scoring":
        feedback_model = ScoreRegressionModel()
        # Load the state dict
        feedback_model.load_state_dict(torch.load(feedback_model_path))
        print("Loading Score Regression model successfully")
    elif feedback_type == "comparing":
        feedback_model = BradleyTerryModel()
        # Load the state dict
        feedback_model.load_state_dict(torch.load(feedback_model_path))
        print("Loading Bradley Terry model successfully")
    elif feedback_type == "ranking":
        feedback_model = RankListNetModel()
        # Load the state dict
        feedback_model.load_state_dict(torch.load(feedback_model_path))
        print("Loading Rank ListNet model successfully")
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
        labels_proba = dataframe["label_proba"].to_numpy()
        labels_binary = dataframe["label_binary"].to_numpy()
        outputs = {
            "features": features,
            "labels_proba": labels_proba,
            "labels_binary": labels_binary
        }
        return outputs
    elif feedback_type == "comparing":
        # Load the training data for Bradley Terry model
        dataframe = pd.read_csv(data_path)
        smiles_1_list = dataframe["smiles_1"].values
        smiles_2_list = dataframe["smiles_2"].values
        features_1 = np.array([compute_fingerprints(smiles) for smiles in smiles_list])
        features_2 = np.array([compute_fingerprints(smiles) for smiles in smiles_list])
        labels_proba = dataframe["label_proba"].values
        labels_binary = dataframe["label_binary"].values
        outputs = {
            "features_1": features_1,
            "features_2": features_2,
            "labels_proba": labels_proba,
            "labels_binary": labels_binary
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
        labels_1_proba = dataframe["label_1_proba"].values
        labels_2_proba = dataframe["label_2_proba"].values
        labels_3_proba = dataframe["label_3_proba"].values
        labels_1_rank = dataframe["label_1_rank"].values
        labels_2_rank = dataframe["label_2_rank"].values
        labels_3_rank = dataframe["label_3_rank"].values
        outputs = {
            "features_1": features_1,
            "features_2": features_2,
            "features_3": features_3,
            "labels_1_proba": labels_1_proba,
            "labels_2_proba": labels_2_proba,
            "labels_3_proba": labels_3_proba,
            "labels_1_rank": labels_1_rank,
            "labels_2_rank": labels_2_rank,
            "labels_3_rank": labels_3_rank
        }
        return outputs
    else:
        raise ValueError("Invalid model type")
    




def extract_ECFP_dataset(init_train_set_path, num_train_samples):
    """
        Load background training data used to pre-train the predictive model    
    """
    
    print("Loading D0")
    train_set = pd.read_csv(init_train_set_path)
    feature_cols = [f"bit{i}" for i in range(2048)]
    target_col = ["activity"]
    smiles_train = train_set["smiles"].values.reshape(-1)
    x_train = train_set[feature_cols].values
    y_train = train_set[target_col].values.reshape(-1)
    sample_weight = np.array([1. for i in range(len(x_train))])
    print("The feature matrix shape: ", x_train.shape)
    print("The labels shape: ", y_train.shape)

    train_sample = train_set[train_set["activity"] == 1].sample(num_train_samples).smiles.tolist()
    return x_train, y_train, sample_weight, smiles_train, train_sample

def read_scaffold_result(scaffold_memory_path, threshold=0.5):
    scaffold_df = pd.read_csv(scaffold_memory_path)

    data_len = len(scaffold_df)
    print("Number of SMILES in scaffold_memory.csv: ", data_len)
    colnames = scaffold_df.columns 
    scaffold = scaffold_df["Scaffold"].to_numpy()
    smiles = scaffold_df["SMILES"].to_numpy()
    human_component = scaffold_df["Human-Component"].to_numpy()
    raw_human_component = scaffold_df["raw_Human-Component"].to_numpy()
    total_score = scaffold_df["total_score"].to_numpy()

    # save the indexes of high scoring molecules for bioactivity
    high_scoring_idx = np.where(total_score > threshold)[0]

    print(f'{len(high_scoring_idx)}/{data_len} high-scoring (> {threshold}) molecules')

    # Only analyse highest scoring molecules
    smiles_high_score = smiles[high_scoring_idx]
    scaffold_high_score = scaffold[high_scoring_idx]
    human_component_high_score = human_component[high_scoring_idx]
    raw_human_component_high_score = raw_human_component[high_scoring_idx]
    total_score_high_score = total_score[high_scoring_idx]
    
    # print shape
    print("Scaffold shape: ", scaffold_high_score.shape)
    print("SMILES shape: ", smiles_high_score.shape)
    print("Human component shape: ", human_component_high_score.shape)
    print("Raw human component shape: ", raw_human_component_high_score.shape)
    print("Total score shape: ", total_score_high_score.shape)
    
    output_high_score = {
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
        
def smiles_human_score(output_high_score, sigma_noise=0.1):
    scaffold_output = output_high_score["scaffold"]
    smiles_output = output_high_score["smiles"]
    human_component_output = output_high_score["human_component"]
    raw_human_component_output = output_high_score["raw_human_component"]
    total_score_output = output_high_score["total_score"]
    drd2_oracle_model = Oracle(name = 'DRD2')
    smiles_score_human = [human_score(drd2_oracle_model, smile, sigma_noise) for smile in smiles_output] 
    return smiles_score_human

def retrain_model(feedback_type, data_outputs, model):
    """
        Retrain the model with the new data
    """
    if feedback_type == "scoring":
        model.train(data_outputs["x_train"], data_outputs["y_train"], data_outputs["sample_weight"])
    elif feedback_type == "comparing":
        model.train(data_outputs["x_train"], data_outputs["y_train"], data_outputs["sample_weight"])
    elif feedback_type == "ranking":
        model.train(data_outputs["x_train"], data_outputs["y_train"], data_outputs["sample_weight"])
    else:
        raise ValueError("Invalid model type")
    return model

def combine_dataset(feedback_type, outputs_1, outputs_2):
    """
        Combine the dataset for training the model
    """
    if feedback_type == "scoring":
        features_1 = outputs_1["features"]
        labels_proba_1 = outputs_1["labels_proba"]
        labels_binary_1 = outputs_1["labels_binary"]
        features_2 = outputs_2["features"]
        labels_proba_2 = outputs_2["labels_proba"]
        labels_binary_2 = outputs_2["labels_binary"]
        features = np.concatenate((features_1, features_2), axis=0)
        labels_proba = np.concatenate((labels_proba_1, labels_proba_2), axis=0)
        labels_binary = np.concatenate((labels_binary_1, labels_binary_2), axis=0)
        outputs = {
            "features": features,
            "labels_proba": labels_proba,
            "labels_binary": labels_binary
        }
        return outputs
    elif feedback_type == "comparing":
        features_1_1 = outputs_1["features_1"]
        features_2_1 = outputs_1["features_2"]
        labels_proba_1 = outputs_1["labels_proba"]
        labels_binary_1 = outputs_1["labels_binary"]
        features_1_2 = outputs_2["features_1"]
        features_2_2 = outputs_2["features_2"]
        labels_proba_2 = outputs_2["labels_proba"]
        labels_binary_2 = outputs_2["labels_binary"]
        features_1 = np.concatenate((features_1_1, features_1_2), axis=0)
        features_2 = np.concatenate((features_2_1, features_2_2), axis=0)
        labels_proba = np.concatenate((labels_proba_1, labels_proba_2), axis=0)
        labels_binary = np.concatenate((labels_binary_1, labels_binary_2), axis=0)
        outputs = {
            "features_1": features_1,
            "features_2": features_2,
            "labels_proba": labels_proba,
            "labels_binary": labels_binary
        }
        return outputs
    elif feedback_type == "ranking":
        features_1_1 = outputs_1["features_1"]
        features_2_1 = outputs_1["features_2"]
        features_3_1 = outputs_1["features_3"]
    return combined_outputs