# Comparison Report: Centralized vs. Federated Face Recognition Attendance System

## Overview
This report compares the baseline **Centralized Learning (CL)** approach with the proposed **Federated Learning (FL)** system for a facial recognition-based attendance application.

## 1. Architectural Comparison

| Component | Centralized Learning (Baseline) | Federated Learning (Proposed) |
| :--- | :--- | :--- |
| **Data Flow** | **Data-to-Global**: Raw face images (cropped/resized) are uploaded to the server via ZIP. | **Model-to-Data**: Raw data stays on the edge device. Only model weights/parameters are shared. |
| **Model Structure** | **Monolithic**: Single MobileFaceNet model (128-dim) trained on all data. | **Hybrid (pFedFace)**: Global Backbone (MobileFaceNet) for feature extraction + Local Head (ArcMargin) for identity. |
| **Training Strategy** | **Bulk Training**: Server trains a single global model once all data is received. | **Iterative Training**: Clients train locally using `FedProx` and aggregate updates via `Flower`. |
| **Privacy** | Low (Server sees all face data). | **High** (Server never sees raw images). |

## 2. Training Parameter Alignment
To ensure the validity of the comparison, both systems follow an identical training configuration:

| Parameter | Centralized Learning (CL) | Federated Learning (FL) | Notes |
| :--- | :--- | :--- | :--- |
| **Backbone Architecture** | MobileFaceNet (128-dim) | MobileFaceNet (128-dim) | Identical for feature validity. |
| **Loss Function** | ArcMarginProduct | ArcMarginProduct | Modern face recognition standard. |
| **Optimizer** | Adam | Adam | Consistency in weight updates. |
| **Learning Rate (LR)** | 1e-4 (E1-21), 5e-5 (E22-36), 1e-5 (E37-45) | 1e-4 (R1-7), 5e-5 (R8-12), 1e-5 (R13-15) | Avoids learning speed bias. |
| **Total Data Iterations** | 45 Epochs | (15 Rounds x 3 Epochs) = 45 Epochs | Equal frequency of seeing data. |
| **Batch Size** | 32 (Total) | 16 (per client) = 32 (Total) | Equivalent gradient load. |
| **Data Augmentation** | Jitter, Rotate, Flip | Jitter, Rotate, Flip | Uniform variation. |
| **Dataset Selection** | Top 50 (Laplacian Var) | Top 50 (Laplacian Var) | Standard input quality. |
| **Label Smoothing** | 0.1 | 0.1 | Prevents model over-confidence. |
| **FedProx (mu)** | N/A | 0.05 | FL-specific to suppress divergence. |
| **Input Resolution** | 112 x 96 (Portrait Squash) | 112 x 96 (Portrait Squash) | Standard MobileFaceNet dimension. |
| **Inference Registry** | Centroid Embedding | Centroid Embedding | Same matching method. |
| **Inference Engine** | **Full PyTorch (CPU)** | **Full PyTorch (CPU)** | Migrated from ONNX for stability. |
| **Lazy Loading** | **Enabled** (at Camera Loop) | **Enabled** (at Camera Loop) | Reduces initial RAM footprint. |
| **Resource Guard** | **Enabled** (Stops Infer during Train) | **Enabled** (Stops Infer during Train) | Prioritize CPU/RAM for training. |
| **Model Versioning** | v0 -> v1 (Single Phase) | v0 -> v1 (After Round 15) | Consistent versioning logic. |
| **MTCNN (Margin 20px)** | Yes | Yes | Unified cropping methodology. |
| **Dataset Split** | 80% Train / 20% Val | 80% Train / 20% Val | Aligned sample distribution. |
| **Inference Buffer** | 10 Frames | 10 Frames | Aligned for stability parity. |

## 3. Key Federated Learning Innovations (Proposed)

The proposed system introduces several advanced FL techniques:

*   **pFedFace (Personalized Federated Face)**: Segregates the "extraction" knowledge (global) from "identity" knowledge (local).
*   **Global BN Merging**: Instead of just averaging weights, the server aggregates **Batch Normalization** statistics (mean/variance) from all clients, significantly stabilizing inference accuracy across different devices.
*   **Knowledge Sharing (Centroids)**: Clients exchange anonymized face centroids. This allows a client to "know" about students from other terminals without seeing their photos, preventing the model from forgetting global identities during local training.
*   **FedProx Optimization**: Uses a proximal term ($\mu$) to account for non-IID (Independent and Identically Distributed) data, which is common in facial recognition (each terminal has different students).

## 4. Workflow Comparison

### Centralized Workflow
1.  **Client**: Preprocesses images -> Packs ZIP -> Uploads to Server.
2.  **Server**: Extracts ZIP -> Trains Model -> Generates Reference Embeddings.
3.  **Client**: Downloads `.pth` model and `reference_embeddings.pth` for inference.

### Federated Workflow (Dynamic Barrier Sync)
1.  **Phase 1: Discovery**: Register student IDs to a Global Map at the server.
2.  **Phase 2: Preprocess**: Local cropping and selection of 50 sharpest images.
3.  **Phase 4: Training**: Parallel training rounds using `Flower` and `FedProx`.
4.  **Phase 5: Registry**: Calculate global BN stats and final centroids for the "World-Knowledge" database.

## 5. Inference & Accuracy
Both systems use **MobileFaceNet** as the backbone and **Cosine Similarity** for matching. To ensure parity, both implement:
*   **MTCNN Square-to-Portrait Prep**: Standard MTCNN crop with 20px margin, squashed to 96x112 portrait.
*   **Temporal Voting**: Uses a shared buffer size of **10 frames** to stabilize predictions.
*   **L2 Normalization**: Applied to both query and reference embeddings.

## Conclusion
The **Federated Learning** proposal offers a privacy-preserving alternative to the Centralized baseline without sacrificing recognition performance. By utilizing **pFedFace**, **Global BN Merging**, and **Knowledge Sharing**, it overcomes the typical challenges of FL in facial recognition, such as identity heterogeneity and model drift.
