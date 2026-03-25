import torch
import torch.nn.functional as F
import numpy as np

def identify_user_globally(query_embedding, local_embeddings_dict, threshold=0.65):
    """
    Identify user based on Cosine Similarity.
    
    Args:
        query_embedding (np.ndarray or torch.Tensor): The embedding of the face to identify.
        local_embeddings_dict (dict): Dictionary mapping user_id to their reference embedding.
        threshold (float): Similarity threshold for 'Unknown'.
        
    Returns:
        tuple: (matched_user_id, confidence)
    """
    if not local_embeddings_dict:
        return "Unknown", 0.0
    
    if isinstance(query_embedding, np.ndarray):
        query_tensor = torch.from_numpy(query_embedding).float()
    else:
        query_tensor = query_embedding.float()
        
    if query_tensor.dim() == 1:
        query_tensor = query_tensor.unsqueeze(0)
        
    # Normalize query embedding
    query_tensor = F.normalize(query_tensor, p=2, dim=1)
    
    best_match = "Unknown"
    max_sim = -1.0
    
    for user_id, ref_embedding in local_embeddings_dict.items():
        if isinstance(ref_embedding, np.ndarray):
            ref_tensor = torch.from_numpy(ref_embedding).float().to(query_tensor.device)
        else:
            ref_tensor = ref_embedding.float().to(query_tensor.device)
            
        if ref_tensor.dim() == 1:
            ref_tensor = ref_tensor.unsqueeze(0)
            
        # Normalize reference embedding
        ref_tensor = F.normalize(ref_tensor, p=2, dim=1)
        
        # Cosine Similarity
        similarity = F.linear(query_tensor, ref_tensor).item()
        
        if similarity > max_sim:
            max_sim = similarity
            best_match = user_id
            
    if max_sim < threshold:
        return "Unknown", max_sim
        
    return best_match, max_sim
