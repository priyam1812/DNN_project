# ==========================================
# Multimodal Visual Story Reasoning Model
# ==========================================

import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

# --------------------------------------------------
# Image Encoder (ResNet18)
# --------------------------------------------------

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained=True, freeze_backbone=True):
        super().__init__()

        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        """
        x: (B, K, 3, H, W)
        """
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)

        feats = self.backbone(x)              # (B*K, 512, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)
        feats = self.fc(feats)                # (B*K, embed_dim)

        feats = feats.view(B, K, -1)           # (B, K, embed_dim)
        return feats


# --------------------------------------------------
# Text Encoder (BERT)
# --------------------------------------------------

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512, model_name="bert-base-uncased", freeze_encoder=True):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.pooler_output         # (B, 768)
        return self.fc(pooled)                 # (B, embed_dim)


# --------------------------------------------------
# Cross-Modal Attention (Innovation)
# --------------------------------------------------

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, image_feats, text_feats):
        """
        image_feats: (B, K, embed_dim)
        text_feats:  (B, embed_dim)
        """
        text_feats = text_feats.unsqueeze(1)   # (B, 1, embed_dim)

        fused, attn_weights = self.attn(
            query=image_feats,
            key=text_feats,
            value=text_feats
        )

        return fused, attn_weights


# --------------------------------------------------
# Temporal Sequence Model (LSTM)
# --------------------------------------------------

class TemporalSequenceModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        """
        x: (B, K, embed_dim)
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]                         # (B, embed_dim)


# --------------------------------------------------
# Decoders
# --------------------------------------------------

class ImageDecoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.fc(x)                      # (B, embed_dim)


class TextDecoder(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=30522):
        super().__init__()
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)                      # (B, vocab_size)


# --------------------------------------------------
# Full Multimodal Model
# --------------------------------------------------

class VisualStoryModel(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        vocab_size=30522,
        cnn_pretrained=True,
        freeze_cnn=True,
        bert_model="bert-base-uncased",
        freeze_bert=True,
        num_attention_heads=4
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim,
            pretrained=cnn_pretrained,
            freeze_backbone=freeze_cnn
        )

        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            model_name=bert_model,
            freeze_encoder=freeze_bert
        )

        self.cross_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads
        )

        self.temporal_model = TemporalSequenceModel(
            embed_dim=embed_dim
        )

        self.image_decoder = ImageDecoder(
            embed_dim=embed_dim
        )

        self.text_decoder = TextDecoder(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        )

    def forward(self, images, input_ids, attention_mask):
        """
        images:         (B, K, 3, 224, 224)
        input_ids:      (B, L)
        attention_mask: (B, L)
        """

        img_feats = self.image_encoder(images)                       # (B, K, D)
        txt_feats = self.text_encoder(input_ids, attention_mask)     # (B, D)

        fused_feats, attn_weights = self.cross_attention(
            img_feats, txt_feats
        )                                                             # (B, K, D)

        story_repr = self.temporal_model(fused_feats)                # (B, D)

        pred_img_embed = self.image_decoder(story_repr)              # (B, D)
        pred_txt_logits = self.text_decoder(story_repr)              # (B, V)

        return pred_img_embed, pred_txt_logits, attn_weights
