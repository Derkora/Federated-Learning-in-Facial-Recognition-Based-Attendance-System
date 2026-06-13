def weighted_average(metrics: list) -> dict:
    # Mengambil jumlah sampel per klien
    examples = [m[0] for m in metrics]
    total_examples = sum(examples)
    if total_examples == 0:
        return {}
    
    # Rata-rata tertimbang metrik pelatihan & validasi
    return {
        "accuracy": sum([
            m[1].get("accuracy", 0.0) * m[0] for m in metrics
        ]) / total_examples,
        "loss": sum([
            m[1].get("loss", 0.0) * m[0] for m in metrics
        ]) / total_examples,
        "val_accuracy": sum([
            m[1].get("val_accuracy", 0.0) * m[0] for m in metrics
        ]) / total_examples,
        "val_loss": sum([
            m[1].get("val_loss", 0.0) * m[0] for m in metrics
        ]) / total_examples,
    }
