"""
embedder.py
-----------
Fine-grained fashion embedding pipeline.

Strategy:
  1. Visual embedding  : Marqo-FashionSigLIP (fine-tuned CLIP for fashion,
                         trained with Generalised Contrastive Learning across
                         7 fashion aspects — colour, material, style, details…)
  2. Colour embedding  : HSV colour histogram (32 bins × 3 channels = 96-dim)
                         provides an explicit, hard colour signal that pure
                         CLIP embeddings can underweight on uniform garments.
  3. Fusion            : weighted concatenation → L2-normalised fused vector
                         (visual weight 0.85, colour weight 0.15)
  4. Query augmentation: average of 2 crops (original + centre crop) at
                         query time for slightly more stable retrieval.
"""

from __future__ import annotations

import io
import logging
from typing import List

import numpy as np
import open_clip
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"
VISUAL_WEIGHT: float = 0.85
COLOR_WEIGHT: float = 0.15
COLOR_BINS: int = 32  # per channel → 96-dim colour histogram


# ---------------------------------------------------------------------------
# Helper: pure-numpy RGB → HSV  (avoids opencv dependency)
# ---------------------------------------------------------------------------
def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 3) uint8 RGB array to an (H, W, 3) float32 HSV array.
    H ∈ [0, 1), S ∈ [0, 1], V ∈ [0, 1].
    """
    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    # Value
    v = maxc

    # Saturation
    s = np.where(maxc > 0, diff / (maxc + 1e-8), 0.0)

    # Hue
    rc = np.where(diff > 0, (maxc - r) / (diff + 1e-8), 0.0)
    gc = np.where(diff > 0, (maxc - g) / (diff + 1e-8), 0.0)
    bc = np.where(diff > 0, (maxc - b) / (diff + 1e-8), 0.0)

    h = np.where(
        maxc == r, (bc - gc) % 6.0,
        np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FashionEmbedder:
    """Loads Marqo-FashionSigLIP and produces fused image embeddings."""

    def __init__(self) -> None:
        logger.info("Loading Marqo-FashionSigLIP model …")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_ID
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        logger.info("Model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_for_index(self, image: Image.Image) -> np.ndarray:
        """
        Single, deterministic embedding for indexing dataset images.
        No augmentation — fast, reproducible.
        """
        visual = self._visual_embedding(image)
        colour = self._colour_histogram(image)
        return self._fuse(visual, colour)

    def embed_query(self, image: Image.Image) -> np.ndarray:
        """
        Multi-crop embedding for query images at search time.
        Averages embeddings from the original image + a tighter centre crop
        for slightly improved retrieval stability.
        """
        crops: List[Image.Image] = [image, self._centre_crop(image)]
        embs = [self._fuse(self._visual_embedding(c), self._colour_histogram(c)) for c in crops]
        avg = np.mean(embs, axis=0).astype(np.float32)
        avg /= np.linalg.norm(avg) + 1e-8
        return avg

    @staticmethod
    def load_image(source: str) -> Image.Image:
        """Load a PIL Image from a local path or HTTP(S) URL."""
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, timeout=15)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return Image.open(source).convert("RGB")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _visual_embedding(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
        return feat.cpu().numpy()[0].astype(np.float32)

    def _colour_histogram(self, image: Image.Image) -> np.ndarray:
        """96-dim normalised HSV histogram (32 bins per channel)."""
        img_np = np.array(image.resize((128, 128)).convert("RGB"))
        hsv = _rgb_to_hsv(img_np)
        hists = []
        ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        for ch, (lo, hi) in enumerate(ranges):
            h, _ = np.histogram(hsv[:, :, ch], bins=COLOR_BINS, range=(lo, hi))
            h = h.astype(np.float32)
            h /= h.sum() + 1e-8
            hists.append(h)
        return np.concatenate(hists).astype(np.float32)

    @staticmethod
    def _fuse(visual: np.ndarray, colour: np.ndarray) -> np.ndarray:
        v = visual / (np.linalg.norm(visual) + 1e-8)
        c = colour / (np.linalg.norm(colour) + 1e-8)
        fused = np.concatenate([VISUAL_WEIGHT * v, COLOR_WEIGHT * c])
        fused /= np.linalg.norm(fused) + 1e-8
        return fused.astype(np.float32)

    @staticmethod
    def _centre_crop(image: Image.Image) -> Image.Image:
        w, h = image.size
        m = min(w, h)
        left = (w - m) // 2
        top = (h - m) // 2
        return image.crop((left, top, left + m, top + m))
