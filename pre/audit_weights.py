import torch
import sys
import os

# Try to import MobileFaceNet from both locations to compare
try:
    # Add prep path to sys.path to find MobileFaceNet from isolated env
    sys.path.append(os.path.abspath("pre/uji-fl-isolated"))
    # Also add federated-learning path
    sys.path.append(os.path.abspath("federated-learning/client/app"))
    
    from utils.mobilefacenet import MobileFaceNet
    print("[INIT] Imported MobileFaceNet from federated-learning/client/app/utils/mobilefacenet.py")
except Exception as e:
    print(f"[ERROR] Could not import MobileFaceNet: {e}")
    sys.exit(1)

def is_shared_param(name):
    n = name.lower()
    if any(x in n for x in ["bn", "running_", "num_batches_tracked"]):
        return False
    return any(x in n for x in ["weight", "bias"])

model = MobileFaceNet()
state_dict = model.state_dict()

all_keys = list(state_dict.keys())
shared_keys = [k for k in all_keys if is_shared_param(k)]
bn_keys = [k for k in all_keys if not is_shared_param(k)]

print(f"\n============================================================")
print(f"MODEL PARAMETER AUDIT (MobileFaceNet)")
print(f"============================================================")
print(f"Total State Dict Keys: {len(all_keys)}")
print(f"Shared Keys (Conv/Linear only): {len(shared_keys)}")
print(f"Batch Normalization Keys: {len(bn_keys)}")
print(f"============================================================")

print("\nSHARED KEYS (First 10):")
for k in shared_keys[:10]:
    print(f"  - {k} (shape: {list(state_dict[k].shape)})")

print("\nBN KEYS (First 10):")
for k in bn_keys[:10]:
    print(f"  - {k} (shape: {list(state_dict[k].shape)})")

if len(shared_keys) == 173:
    print("\n✅ MATCH: Shared parameter count is exactly 173.")
else:
    print(f"\n❌ MISMATCH: Found {len(shared_keys)} shared parameters (expected 173).")

# Check original global model if available
model_v0_path = "federated-learning/server/app/model/global_model_v0.pth"
if os.path.exists(model_v0_path):
    print(f"\nChecking weights file at {model_v0_path}...")
    try:
        loaded = torch.load(model_v0_path, map_location="cpu")
        if isinstance(loaded, dict):
            print(f"  - File is a dict with {len(loaded)} keys.")
            loaded_shared = [k for k in loaded.keys() if is_shared_param(k)]
            print(f"  - File contains {len(loaded_shared)} shared parameters.")
        elif isinstance(loaded, list):
            print(f"  - File is a list with {len(loaded)} elements.")
    except Exception as e:
        print(f"  - Error loading weights file: {e}")
