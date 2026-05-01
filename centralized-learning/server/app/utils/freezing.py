import torch.nn as nn

def set_model_freeze(model, freeze_mode="early"):
    """
    Mengontrol pembekuan lapisan (Partial Freezing) pada MobileFaceNet.
    
    Args:
        model (nn.Module): Instance dari MobileFaceNet.
        freeze_mode (str): Opsi mode pembekuan:
            - "none": Tidak ada yang dibekukan (semua parameter dilatih).
            - "early": Membekukan Conv1 sampai bottleneck stage 2 (conv_3).
            - "backbone": Membekukan seluruh backbone MobileFaceNet.
    """
    # 1. Aktifkan semua parameter terlebih dahulu (Reset)
    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == "none":
        print("[FREEZE] Mode: 'none'. Seluruh backbone akan dilatih.")
        return

    if freeze_mode == "early":
        print("[FREEZE] Mode: 'early'. Membekukan Conv1 hingga Stage 2 (conv_3).")
        # Bekukan Conv1 dan dw_conv1 (Awal)
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.dw_conv1.parameters():
            param.requires_grad = False
            
        # Bekukan blocks (Bottlenecks)
        # Berdasarkan Mobilefacenet_bottleneck_setting:
        # Index 0-4: Stage 1 (conv_23)
        # Index 5-11: Stage 2 (conv_34) -> Akhir dari "early layers" (conv_3)
        # Index 12-14: Stage 3 (conv_45) -> Ini tetap AKTIF (Late layers)
        
        for i in range(12): # Bekukan 12 bottleneck pertama (0 s/d 11)
            for param in model.blocks[i].parameters():
                param.requires_grad = False
                
    elif freeze_mode == "backbone":
        print("[FREEZE] Mode: 'backbone'. Seluruh backbone MobileFaceNet dibekukan.")
        for param in model.parameters():
            param.requires_grad = False
