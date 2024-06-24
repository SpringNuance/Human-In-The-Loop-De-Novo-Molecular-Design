# Scripts for writing and modifying configuration json files of REINVENT

import os
import json

def write_REINVENT_config_bradley_terry(reinvent_dir, reinvent_env, output_dir, conf_filename, jobid, jobname):

    diversity_filter = {
        "name": "IdenticalMurckoScaffold",
        "bucket_size": 25,
        "minscore": 0.2,
        "minsimilarity": 0.4
    }

    inception = {
        "memory_size": 20,
        "sample_size": 5,
        "smiles": []
    }

    human_component = {
        "component_type": "predictive_property",
        "name": "Human-Component",
        "weight": 1,
        "specific_parameters": {
            "model_path": "/home/springnuance/reinvent-hitl/Base-Code-Binh/training_Bradley_Terry_model/bradley_terry_model.pth",
            "model_name": "bradley_terry",
            "bradley_terry": "classification",
            "descriptor_type": "ecfp",
            "size": 2048,
            "container_type": "bradley_terry_container",
            "use_counts": True,
            "use_features": True,
            "transformation": {
                "transformation_type": "no_transformation",
            }
        }
    }

    scoring_function = {
        "name": "custom_sum",
        "parallel": False,
        "parameters": [
            human_component
        ]
    }

    configuration = {
        "version": 3,
        "run_type": "reinforcement_learning",
        "model_type": "default",
        "parameters": {
            "scoring_function": scoring_function
        }
    }

    configuration["parameters"]["diversity_filter"] = diversity_filter
    configuration["parameters"]["inception"] = inception

    configuration["parameters"]["reinforcement_learning"] = {
        "prior": os.path.join(reinvent_dir, "data/random.prior.new"),
        "agent": os.path.join(reinvent_dir, "data/random.prior.new"),
        "n_steps": 250,
        "sigma": 128,
        "learning_rate": 0.0001,
        "batch_size": 128,
        "reset": 0,
        "reset_score_cutoff": 0.5,
        "margin_threshold": 50
    }

    configuration["logging"] = {
        "sender": "http://127.0.0.1",
        "recipient": "local",
        "logging_frequency": 0,
        "logging_path": os.path.join(output_dir, "progress.log"),
        "result_folder": os.path.join(output_dir, "results"),
        "job_name": jobname,
        "job_id": jobid
    }

    # write the configuration file to disc
    configuration_JSON_path = os.path.join(output_dir, conf_filename)
    with open(configuration_JSON_path, 'w') as f:
        json.dump(configuration, f, indent=4, sort_keys=True)
    
    return configuration_JSON_path


def write_sample_file(jobid, jobname, agent_dir, N):
  configuration={
    "logging": {
        "job_id": jobid,
        "job_name":  "sample_agent_{}".format(jobname),
        "logging_path": os.path.join(agent_dir, "sampling.log"),
        "recipient": "local",
        "sender": "http://127.0.0.1"
    },
    "parameters": {
        "model_path": os.path.join(agent_dir, "Agent.ckpt"),
        "output_smiles_path": os.path.join(agent_dir, "sampled_N_{}.csv".format(N)),
        "num_smiles": N,
        "batch_size": 128,                          
        "with_likelihood": False
    },
    "run_type": "sampling",
    "version": 2
  }
  conf_filename = os.path.join(agent_dir, "evaluate_agent_config.json")
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  return conf_filename
