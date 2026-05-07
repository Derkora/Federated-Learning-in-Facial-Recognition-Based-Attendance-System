import torch
import torch.nn.functional as F
import numpy as np

from app.utils.logging import get_logger

def identify_user_globally(query_embedding, local_embeddings_dict, threshold=0.35, metric="cosine", verbose=True):
    logger = get_logger()
    if not local_embeddings_dict:
        return "Unknown", 0.0
    
    # 1. Siapkan Tensor Query (1, 128)
    if isinstance(query_embedding, np.ndarray):
        query_tensor = torch.from_numpy(query_embedding).float()
    else:
        query_tensor = query_embedding.float()
        
    if query_tensor.dim() == 1:
        query_tensor = query_tensor.unsqueeze(0)
    
    # Normalisasi L2 Query
    query_tensor = F.normalize(query_tensor, p=2, dim=1)
    
    # 2. VEKTORISASI: Konversi semua referensi ke satu matriks (N, 128)
    user_ids = list(local_embeddings_dict.keys())
    ref_list = []

    for uid in user_ids:
        ref = local_embeddings_dict[uid]
        if isinstance(ref, np.ndarray):
            ref = torch.from_numpy(ref).float()
        else:
            ref = ref.float()
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        ref_list.append(ref)
    
    # Gabungkan semua (N, 128)
    ref_matrix = torch.cat(ref_list, dim=0)
    ref_matrix = F.normalize(ref_matrix, p=2, dim=1)
    
    # 3. Hitung Skor Sekaligus
    if metric == "cosine":
        scores = torch.mm(query_tensor, ref_matrix.t())
        # Ambil Top 2 untuk riset (Tanpa Blocker)
        vals, idxs = torch.topk(scores, k=min(2, len(user_ids)), dim=1)
        
        confidence = float(vals[0][0].item())
        best_match = user_ids[idxs[0][0].item()]
        
        # Logging Tambahan untuk Riset (NRP 028 vs 072 etc)
        if len(user_ids) > 1 and verbose:
            second_match = user_ids[idxs[0][1].item()]
            second_conf = float(vals[0][1].item())
            logger.info(f"Riset: Top1: {best_match} ({confidence:.3f}) | Top2: {second_match} ({second_conf:.3f}) | Gap: {confidence-second_conf:.3f}")
    else:
        # Euclidean Distance
        dists = torch.cdist(query_tensor, ref_matrix, p=2)
        min_dist, min_idx = torch.min(dists, dim=1)
        confidence = float(1.0 - (min_dist.item() / 2.0))
        best_match = user_ids[min_idx.item()]

    # 4. Ambang Batas
    if confidence < threshold:
        if verbose:
            logger.info(f"Match '{best_match}' rejected (Score: {confidence:.3f} < threshold {threshold})")
        return "Unknown", confidence
        
    if verbose:
        logger.success(f"Match '{best_match}' accepted (Score: {confidence:.3f})")

    return best_match, confidence
