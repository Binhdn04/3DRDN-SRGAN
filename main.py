import yaml
import os
from src.models import get_model
from src.training import get_trainer

if __name__ == "__main__":
    os.makedirs("checkpoints/tensor_checkpoints/generator_checkpoint/G", exist_ok=True)
    os.makedirs("checkpoints/tensor_checkpoints/discriminator_checkpoint/Y", exist_ok=True)
    os.makedirs("results/plots/training", exist_ok=True)
    os.makedirs("results/plots/validation", exist_ok=True)
    
    config = yaml.safe_load(open("configs/default.yaml"))
    
    trainer = get_trainer(config["training"]["version"])
    trainer(None, config)