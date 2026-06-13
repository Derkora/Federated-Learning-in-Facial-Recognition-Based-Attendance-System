import os
import json

def select_best_faces(self, folder_path, n=50):
    if not os.path.exists(folder_path):
        return []
            
    final_cache_path = os.path.join(
        folder_path, ".selection_cache.json"
    )
    scores_cache_path = os.path.join(
        folder_path, ".laplacian_scores.json"
    )
 
    # Gunakan cache hasil seleksi sebelumnya jika ada
    if os.path.exists(final_cache_path):
        try:
            with open(final_cache_path, "r") as f:
                cache_data = json.load(f)
                if cache_data.get("n") == n:
                    return cache_data.get("filenames", [])
        except Exception:
            pass

    all_imgs = sorted([
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if not all_imgs: 
        return []
 
    scored_map = {}
    if os.path.exists(scores_cache_path):
        try:
            with open(scores_cache_path, "r") as f:
                scored_map = json.load(f)
        except Exception:
            pass
 
    needs_save = False
    for img_name in all_imgs:
        if img_name in scored_map:
            continue
            
        img_path = os.path.join(folder_path, img_name)
        scored_map[img_name] = self.get_blur_score(img_path)
        needs_save = True
 
    if needs_save:
        try:
            with open(scores_cache_path, "w") as f:
                json.dump(scored_map, f)
        except Exception:
            pass
     
    # Urutkan dan pilih N foto tertajam
    scored_list = [
        (name, score) for name, score in scored_map.items()
    ]
    scored_list.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in scored_list[:n]]
    
    # Simpan hasil seleksi ke cache
    try:
        with open(final_cache_path, "w") as f:
            json.dump({"n": n, "filenames": selected}, f)
    except Exception:
        pass
 
    return selected
