# src/models/__init__.py
from .model_v1 import Model3DRDN as ModelV1
from .model_v2 import Model3DRDN as ModelV2
from .model_v3 import Model3DRDN as ModelV3  

def get_model(version, **kwargs):
    models = {
        "v1": ModelV1,
        "v2": ModelV2,  
        "v3": ModelV3   
    }
    if version not in models:
        raise ValueError(f"Model version {version} not supported. Available versions: {list(models.keys())}")
    return models[version](**kwargs)