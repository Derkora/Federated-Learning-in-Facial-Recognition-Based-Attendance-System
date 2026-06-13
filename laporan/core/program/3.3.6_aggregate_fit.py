import torch
import flwr as fl

def aggregate_fit(
    self, server_round: int, results: list, failures: list
):
    # Agregasi parameter terbobot Flower
    aggregated_parameters, aggregated_metrics = (
        super().aggregate_fit(
            server_round, results, failures
        )
    )
    if aggregated_parameters is None:
        return None, aggregated_metrics
 
    # Urutkan bobot hasil agregasi ke NumPy array
    params_np = (
        fl.common.parameters_to_ndarrays(
            aggregated_parameters
        )
    )
    sd = self.load_previous_global_weights()
    all_keys = list(sd.keys())
    
    # Filter lapisan konvolusi (tanpa BN) untuk pFedFace
    bn_patterns = ['bn', 'running_', 'num_batches_tracked']
    conv_keys = [
        k for k in all_keys 
        if not any(x in k.lower() for x in bn_patterns)
    ]
    
    # Tentukan kunci target parameter yang akan diperbarui
    target_keys = (
        conv_keys if len(params_np) == len(conv_keys) 
        else all_keys
    )
    
    # Rekonstruksi parameter ke Tensor PyTorch
    backbone_params = {}
    bn_params = {}
    for i, k in enumerate(target_keys):
        if i < len(params_np):
            val = torch.from_numpy(params_np[i].copy())
            if any(x in k.lower() for x in bn_patterns):
                bn_params[k] = val
            else:
                backbone_params[k] = val
    
    # Perbarui bobot model global baru
    sd.update(backbone_params)
    if bn_params:
        sd.update(bn_params)
    
    # Simpan bobot teragregasi ke database
    self.save_new_global_weights(sd)
    return aggregated_parameters, aggregated_metrics
