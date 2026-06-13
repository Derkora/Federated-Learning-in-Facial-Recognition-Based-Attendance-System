def set_model_freeze(model, freeze_mode="early"):
    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == "none":
        return

    # Bekukan lapisan input awal MobileFaceNet
    if freeze_mode == "early":
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.dw_conv1.parameters():
            param.requires_grad = False
        
        # Bekukan 12 blok bottleneck awal pada backbone
        for i in range(12): 
            for param in model.blocks[i].parameters():
                param.requires_grad = False
        
    # Bekukan seluruh parameter backbone
    elif freeze_mode == "backbone":
        for param in model.parameters():
            param.requires_grad = False
