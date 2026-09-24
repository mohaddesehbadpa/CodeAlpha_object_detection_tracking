"""
person_reid.py
Lightweight appearance-based person re-identification.

How it works:
- We take a reference photo of the person you're looking for.
- We run it through a pretrained ResNet18 (ImageNet weights) with the final
  classification layer removed, which gives us a 512-dim "appearance
  embedding" - a numeric fingerprint of how that person looks (clothing
  color, build, etc.).
- For every person the tracker detects in the video, we crop their bounding
  box and compute the same kind of embedding.
- We compare embeddings with cosine similarity. A high score (> threshold)
  means "this is probably the same person."

This is NOT face recognition - it works even when the face isn't clearly
visible, using overall appearance instead. That also means it can be
fooled by two people wearing near-identical outfits. For a stronger
result you could swap this out for a proper re-ID backbone (e.g. OSNet
via the `torchreid` package) without changing the rest of the app.
"""

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


class PersonReID:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = torch.nn.Identity()  # strip classifier, keep 512-dim features
        self.model = base.eval().to(self.device)
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, crop_bgr):
        """crop_bgr: an OpenCV BGR image (a cropped bounding box)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).cpu().numpy().flatten()
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

    @staticmethod
    def cosine_similarity(a, b):
        if a is None or b is None:
            return -1.0
        return float(np.dot(a, b))
