import json
import os

def get_or_create_model_version(learning_type: str, epochs: int, rounds: int = None, dataset: str = "students"):
    version_num = 1
    if learning_type == "cl":
        version_str = f"cl_{dataset}_{epochs}e"
    else:
        version_str = f"fl_{dataset}_{rounds}r_{epochs}e"
    is_new = True
    return version_num, version_str, is_new
