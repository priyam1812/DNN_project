
# ==========================================
# Utility Functions & Dataset Loader
# ==========================================

import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_story_text(text: str) -> str:
    """
    Remove StoryReasoning markup tags such as <gdi>, <gdo>, etc.
    """
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# StoryReasoning Dataset (RAM-safe)
# --------------------------------------------------

class StoryReasoningDataset(Dataset):
    """
    RAM-safe and PIL-safe dataset loader for StoryReasoning.
    Avoids EXIF decoding issues on Windows.
    """

    def __init__(
        self,
        hf_dataset_name: str,
        split: str = "train",
        max_stories: int = 200,
        sequence_length: int = 4,
        image_size=(224, 224),
        text_max_length: int = 128
    ):
        super().__init__()

        self.sequence_length = sequence_length

        # Load dataset WITHOUT automatic decoding
        self.dataset = load_dataset(
            hf_dataset_name,
            split=split,
            decode=False    # 🔥 CRITICAL FIX
        )

        if max_stories is not None:
            self.dataset = self.dataset.select(range(max_stories))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        # Build index map safely (no iteration over decoded images)
        self.index_map = []
        for i in range(len(self.dataset)):
            frame_count = self.dataset[i]["frame_count"]
            for j in range(frame_count - sequence_length):
                self.index_map.append((i, j))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        story_idx, start = self.index_map[idx]
        example = self.dataset[story_idx]

        # Load images manually (PIL-safe)
        images = [
            img.convert("RGB")
            for img in example["images"]
        ]

        story_text = clean_story_text(example["story"])

        input_images = images[start:start + self.sequence_length]
        target_image = images[start + self.sequence_length]

        input_imgs = [self.transform(img) for img in input_images]
        target_img = self.transform(target_image)

        return (
            torch.stack(input_imgs),   # (K, 3, H, W)
            story_text,
            target_img,
            story_text
        )


# --------------------------------------------------
# Evaluation Utilities
# --------------------------------------------------

def text_topk_accuracy(logits, target_tokens, k=1):
    """
    Compute Top-k accuracy for text prediction.
    """
    topk_preds = logits.topk(k, dim=1).indices
    correct = (topk_preds == target_tokens.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def image_cosine_similarity(pred_embed, target_embed):
    """
    Compute cosine similarity between predicted and target image embeddings.
    """
    return torch.nn.functional.cosine_similarity(
        pred_embed, target_embed
    ).mean().item()
