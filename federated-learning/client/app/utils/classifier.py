import torch
import torch.nn.functional as F
import numpy as np

def identify_user_globally(query_embedding, local_embeddings_dict, threshold=0.35, metric="cosine"):
    if not local_embeddings_dict:
        return "Unknown", 0.0
    
    # Siapkan Tensor Query
    query_tensor = torch.from_numpy(query_embedding).float() if isinstance(query_embedding, np.ndarray) else query_embedding.float()
    if query_tensor.dim() == 1:
        query_tensor = query_tensor.unsqueeze(0)
    
    # Paksa Normalisasi L2 (Kritis untuk Cosine dan Euclidean ternormalisasi)
    query_tensor = F.normalize(query_tensor, p=2, dim=1)
    
    best_match = "Unknown"
    max_sim = -1.0
    min_dist = 10.0 # Nilai tinggi untuk pencarian Euclidean
    
    for user_id, ref_embedding in local_embeddings_dict.items():
        ref_tensor = torch.from_numpy(ref_embedding).float() if isinstance(ref_embedding, np.ndarray) else ref_embedding.float()
        if ref_tensor.dim() == 1:
            ref_tensor = ref_tensor.unsqueeze(0)
        
        ref_tensor = F.normalize(ref_tensor, p=2, dim=1)
        
        if query_tensor.shape[1] != ref_tensor.shape[1]:
            continue

        if metric == "cosine":
            similarity = F.linear(query_tensor, ref_tensor).item()
            if similarity > max_sim:
                max_sim = similarity
                best_match = user_id
        else:
            dist = torch.dist(query_tensor, ref_tensor).item()
            if dist < min_dist:
                min_dist = dist
                best_match = user_id

    # Perhitungan Kepercayaan & Ambang Batas
    if metric == "cosine":
        confidence = float(max_sim)
    else:
        # Konversi Euclidean [0, 2] menjadi kemiripan [0, 1]
        confidence = float(1.0 - (min_dist / 2.0))
        
    if confidence < threshold:
        print(f"[CLASSIFIER] Match '{best_match}' rejected (Score: {confidence:.3f} < threshold {threshold})")
        return "Unknown", confidence
        
    print(f"[CLASSIFIER] Match '{best_match}' accepted (Score: {confidence:.3f})")
    return best_match, confidence
