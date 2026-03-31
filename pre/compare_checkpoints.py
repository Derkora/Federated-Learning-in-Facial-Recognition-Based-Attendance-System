import torch
import os

def check_pth(path, label):
    if not os.path.exists(path):
        print(f"[MISSING] {label} not found at {path}")
        return None
    
    try:
        data = torch.load(path, map_location="cpu")
        print(f"\n[{label}] Path: {path}")
        
        if isinstance(data, dict):
            print(f"  - Type: Dict")
            print(f"  - Total Keys: {len(data)}")
            # Filter shared
            shared = [k for k in data.keys() if not any(x in k.lower() for x in ["bn", "running_", "num_batches"])]
            print(f"  - Shared Keys (approx): {len(shared)}")
            return set(data.keys())
        elif isinstance(data, list):
            print(f"  - Type: List (NDArrays)")
            print(f"  - Total Elements: {len(data)}")
            return len(data)
        else:
            print(f"  - Type: {type(data)}")
            return None
    except Exception as e:
        print(f"  - Error: {e}")
        return None

print("============================================================")
print("CHECKPOINT ARCHITECTURE COMPARISON")
print("============================================================")

# 1. Isolated Checkpoint (from validate_fl.py)
iso_path = "pre/uji-fl-isolated/test_backbone.pth"
iso_keys = check_pth(iso_path, "ISOLATED")

# 2. Server Data Checkpoint
srv_path = "federated-learning/server/app/data/backbone.pth"
srv_keys = check_pth(srv_path, "DOCKER (SERVER)")

# 3. Client Models Checkpoint
cli_path = "federated-learning/client/app/artifacts/models/backbone.pth"
cli_keys = check_pth(cli_path, "DOCKER (CLIENT)")

print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)

if iso_keys and srv_keys:
    if isinstance(iso_keys, set) and isinstance(srv_keys, set):
        diff = iso_keys.symmetric_difference(srv_keys)
        if not diff:
            print("✅ ARCHITECTURE MATCH: Both checkpoints have identical keys.")
        else:
            print(f"❌ ARCHITECTURE MISMATCH: {len(diff)} keys differ between them.")
            print(f"Example diff: {list(diff)[:5]}")
    elif isinstance(iso_keys, int) and isinstance(srv_keys, int):
        if iso_keys == srv_keys:
            print(f"✅ COUNT MATCH: Both use {iso_keys} parameters (Numpy format).")
        else:
            print(f"❌ COUNT MISMATCH: {iso_keys} vs {srv_keys}")
    else:
        print("⚠️ Format mismatch (One is dict, one is list). This indicates a protocol difference.")

print("\nNOTE: 173 params is the standard for MobileFaceNet shared conv weights.")
