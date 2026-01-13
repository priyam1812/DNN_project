
# ==========================================
# Visualization Utilities for Results
# ==========================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import pandas as pd


# --------------------------------------------------
# Ensure result directories exist
# --------------------------------------------------

def ensure_result_dirs():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


# --------------------------------------------------
# Plot 1: Training Loss Curve
# --------------------------------------------------

def plot_training_loss(epochs, losses):
    ensure_result_dirs()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Across Epochs")
    plt.grid(True)

    plt.savefig(
        "results/figures/training_loss.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# --------------------------------------------------
# Plot 2: Cross-Modal Attention Heatmap
# --------------------------------------------------

def plot_attention_heatmap(attn_weights):
    """
    attn_weights: (B, K, 1)
    """
    ensure_result_dirs()

    attn = attn_weights[0].cpu().numpy()

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        attn,
        annot=True,
        cmap="viridis",
        cbar=True,
        yticklabels=[f"Timestep {i+1}" for i in range(attn.shape[0])],
        xticklabels=["Text"]
    )

    plt.title("Cross-Modal Attention Weights")

    plt.savefig(
        "results/figures/cross_modal_attention.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# --------------------------------------------------
# Plot 3: Image Embedding Similarity Histogram
# --------------------------------------------------

def plot_image_similarity(pred_embed, target_embed):
    ensure_result_dirs()

    similarity = F.cosine_similarity(pred_embed, target_embed).cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(similarity, bins=20)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Image Embedding Similarity Distribution")

    plt.savefig(
        "results/figures/image_embedding_similarity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# --------------------------------------------------
# Plot 4: Text Top-5 Confidence
# --------------------------------------------------

def plot_text_topk_confidence(logits, k=5):
    ensure_result_dirs()

    probs = torch.softmax(logits, dim=1)
    topk_probs, topk_indices = probs[0].topk(k)

    plt.figure(figsize=(6, 4))
    plt.bar(range(k), topk_probs.cpu().numpy())
    plt.xticks(range(k), [f"Token {i}" for i in topk_indices.cpu().numpy()])
    plt.ylabel("Probability")
    plt.title(f"Top-{k} Text Prediction Confidence")

    plt.savefig(
        "results/figures/text_top5_confidence.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# --------------------------------------------------
# Save Metrics Table
# --------------------------------------------------

def save_metrics_table(metrics_dict):
    ensure_result_dirs()

    df = pd.DataFrame([metrics_dict])
    df.to_csv(
        "results/tables/metrics_summary.csv",
        index=False
    )
