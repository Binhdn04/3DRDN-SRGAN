# src/training/__init__.py
from .trainer_v1 import main_loop as main_loop_v1
from .trainer_v2 import main_loop as main_loop_v2
from .trainer_v3 import main_loop as main_loop_v3

def train_unified(model, config):
    training_config = config.get("training", {})
    version = training_config["version"]
    
    common_params = {
        "EPOCHS": training_config["epochs"],
        "BATCH_SIZE": training_config["batch_size"],
        "EPOCH_START": training_config["epoch_start"],
        "LAMBDA_ADV": training_config["lambda_adv"],
        "LAMBDA_GRD_PEN": training_config["lambda_grad_pen"],
        "CRIT_ITER": training_config["crit_iter"],
        "TRAIN_ONLY": training_config["train_only"],
        "MODEL": training_config["model"]
    }
    
    version_configs = {
        "v1": {
            "main_loop": main_loop_v1,
            "params": {
                "LR": training_config["learning_rate"],
                "DB": training_config["dense_blocks"],
                "DU": training_config["dense_units"],
                **common_params
            }
        },
        "v2": {
            "main_loop": main_loop_v2,
            "params": {
                "LR": training_config["learning_rate"],
                "DB": training_config["dense_blocks"],
                "DU": training_config["dense_units"],
                **common_params
            }
        },
        "v3": {
            "main_loop": main_loop_v3,
            "params": {
                "NO_OF_RDB_BLOCKS": training_config["num_rdb_blocks"],
                "GROWTH_RATE": training_config["growth_rate"],
                "NO_OF_DENSE_LAYERS": training_config["num_dense_layers"],
                "LR_G": training_config["lr_g"],
                "LR_D": training_config["lr_d"],
                "RESIDUAL_SCALING": training_config["residual_scaling"],
                **common_params
            }
        }
    }
    
    if version not in version_configs:
        raise ValueError(f"Training version {version} not supported. Available versions: {list(version_configs.keys())}")
    
    config_data = version_configs[version]
    main_loop_func = config_data["main_loop"]
    params = config_data["params"]
    
    return main_loop_func(**params)

def get_trainer(version):
    return train_unified