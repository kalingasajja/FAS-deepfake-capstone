"""
pipeline/spatial_branch.py
Spatial Artifact Detection Branch using Vision Transformer (SigLIP-Base + MLP).
Source: Adapted directly from Rohith's implementation (origin/Rohith:Capstonev2.ipynb)
"""

import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SpatialDeepfakeDetector(nn.Module):
    """
    SigLIP-Base Vision Transformer with a 2-layer MLP classification head.
    Backbone: google/siglip-base-patch16-224 (768 embedding dimension).
    Head: Linear(768, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)
    """
    def __init__(self, weights_path: str = None, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 1. Load pretrained SigLIP backbone
        from transformers import SiglipVisionModel
        self.backbone = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. MLP Classification Head
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # 3. Load weights if available
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.head.load_state_dict(state_dict)
            print(f"[Spatial Branch] Loaded trained head weights from {weights_path}")
        else:
            print("[Spatial Branch] Initialized SigLIP backbone with default MLP head")

        self.to(self.device)
        self.eval()

        # Standard SigLIP normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return self.head(outputs.pooler_output)

    def predict_image(self, bgr_crop) -> float:
        """
        Runs spatial inference on an aligned BGR face crop.
        Returns continuous probability score p_spatial in [0.0, 1.0] (higher = more likely fake).
        """
        import cv2
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.forward(tensor)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def predict_clip(self, bgr_frames: list) -> float:
        """
        Calculates frame-level spatial probabilities across video frames and returns the average score.
        """
        if not bgr_frames:
            return 0.5
        scores = [self.predict_image(f) for f in bgr_frames]
        return float(sum(scores) / len(scores))
