# ==========================================
# Training Script for Visual Story Reasoning
# ==========================================

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from model import VisualStoryModel
from utils import (
    StoryReasoningDataset,
    set_seed,
    text_topk_accuracy,
    image_cosine_similarity
)

from visualize import (
    plot_training_loss,
    plot_attention_heatmap,
    plot_image_similarity,
    plot_text_topk_confidence,
    save_metrics_table
)

# --------------------------------------------------
# Load configuration
# --------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

set_seed(config["experiment"]["seed"])

device = (
    "cuda"
    if torch.cuda.is_available() and config["experiment"]["device"] == "cuda"
    else "cpu"
)

# --------------------------------------------------
# Dataset & DataLoader
# --------------------------------------------------

train_dataset = StoryReasoningDataset(
    hf_dataset_name=config["dataset"]["hf_dataset"],
    split=config["dataset"]["split"],
    max_stories=config["dataset"]["max_stories"],
    sequence_length=config["dataset"]["sequence_length"],
    image_size=tuple(config["dataset"]["image_size"]),
    text_max_length=config["dataset"]["text_max_length"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = BertTokenizer.from_pretrained(
    config["model"]["text_encoder"]["model_name"]
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = VisualStoryModel(
    embed_dim=config["model"]["embedding_dim"],
    vocab_size=config["model"]["decoders"]["text_decoder"]["vocab_size"],
    cnn_pretrained=config["model"]["visual_encoder"]["pretrained"],
    freeze_cnn=config["model"]["visual_encoder"]["freeze_backbone"],
    bert_model=config["model"]["text_encoder"]["model_name"],
    freeze_bert=config["model"]["text_encoder"]["freeze_encoder"],
    num_attention_heads=config["model"]["fusion"]["num_heads"]
).to(device)

# --------------------------------------------------
# Losses & Optimizer
# --------------------------------------------------

image_loss_fn = nn.MSELoss()
text_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)

# --------------------------------------------------
# Training Loop
# --------------------------------------------------

num_epochs = config["training"]["epochs"]
epoch_losses = []

print("Starting training...")
print(f"Device: {device}")
print(f"Epochs: {num_epochs}")
print("-" * 50)

model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for images, text_batch, tgt_img, _ in train_loader:
        images = images.to(device)
        tgt_img = tgt_img.to(device)

        # Tokenize text
        encoded = tokenizer(
            list(text_batch),
            padding=True,
            truncation=True,
            max_length=config["dataset"]["text_max_length"],
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward pass
        pred_img_embed, pred_txt_logits, attn_weights = model(
            images, input_ids, attention_mask
        )

        # Target image embedding
        with torch.no_grad():
            target_img_embed = model.image_encoder(
                tgt_img.unsqueeze(1)
            ).squeeze(1)

        # Target text token (first token)
        target_tokens = input_ids[:, 0]

        # Loss
        loss_img = image_loss_fn(pred_img_embed, target_img_embed)
        loss_txt = text_loss_fn(pred_txt_logits, target_tokens)

        loss = (
            config["training"]["losses"]["image_loss"]["weight"] * loss_img +
            config["training"]["losses"]["text_loss"]["weight"] * loss_txt
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    epoch_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

print("Training completed.")

# --------------------------------------------------
# Save Model
# --------------------------------------------------

model_path = "results/visual_story_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# --------------------------------------------------
# Generate Visualizations & Metrics (POST-TRAIN)
# --------------------------------------------------

model.eval()

with torch.no_grad():
    images, text_batch, tgt_img, _ = next(iter(train_loader))
    images = images.to(device)
    tgt_img = tgt_img.to(device)

    encoded = tokenizer(
        list(text_batch),
        padding=True,
        truncation=True,
        max_length=config["dataset"]["text_max_length"],
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    pred_img_embed, pred_txt_logits, attn_weights = model(
        images, input_ids, attention_mask
    )

    target_img_embed = model.image_encoder(
        tgt_img.unsqueeze(1)
    ).squeeze(1)

    # Metrics
    top1_acc = text_topk_accuracy(pred_txt_logits, input_ids[:, 0], k=1)
    top5_acc = text_topk_accuracy(pred_txt_logits, input_ids[:, 0], k=5)
    img_similarity = image_cosine_similarity(pred_img_embed, target_img_embed)

# --------------------------------------------------
# Save Plots
# --------------------------------------------------

plot_training_loss(
    epochs=list(range(1, num_epochs + 1)),
    losses=epoch_losses
)

plot_attention_heatmap(attn_weights)
plot_image_similarity(pred_img_embed, target_img_embed)
plot_text_topk_confidence(pred_txt_logits)

# --------------------------------------------------
# Save Metrics Table
# --------------------------------------------------

save_metrics_table({
    "epochs": num_epochs,
    "final_training_loss": epoch_losses[-1],
    "text_top1_accuracy": top1_acc,
    "text_top5_accuracy": top5_acc,
    "image_cosine_similarity": img_similarity
})

print("Visualizations and metrics saved to results/ directory.")
