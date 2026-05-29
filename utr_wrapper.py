"""
utr_wrapper.py
================
Inference wrapper around the locally-hosted UTRNet model for printed Urdu text
recognition. This wraps the *vendored* official UTRNet source (model.py +
modules/ + utils.py, copied from
https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)
so the network architecture matches the released `best_norm_ED.pth` state_dict
exactly.

The preprocessing and decoding here are a faithful port of the repository's
single-image script `read.py`, in particular:

    * grayscale input (input_channel = 1),
    * a LEFT-RIGHT FLIP of the image (Urdu is right-to-left; the model was
      trained on flipped crops),
    * aspect-ratio-preserving resize to height 32, capped at width 400,
    * NormalizePAD (scale to [-1, 1] then right-pad with the border column),
    * CTC greedy decode via the repo's CTCLabelConverter.

UTRNet-Large configuration (from read.py): HRNet feature extractor
(output_channel = 32), DBiLSTM sequence model, CTC head, hidden_size = 256.

Environment variables
---------------------
UTRNET_MODEL_PATH    Path to the state_dict checkpoint.
                     Default: saved_models/UTRNet-Large/best_norm_ED.pth
UTRNET_CHARSET_PATH  Path to the glyph list (one glyph per line).
                     Default: UrduGlyphs.txt
UTRNET_DEVICE        "cuda", "mps", or "cpu". Auto-detected if unset.
"""

from __future__ import annotations

import os
import math
import logging
from types import SimpleNamespace
from typing import List, Optional

import torch
from PIL import Image
import torchvision.transforms as T

# Vendored official UTRNet source.
from model import Model
from utils import CTCLabelConverter

from line_segmenter import segment_lines

logger = logging.getLogger("utrnet")

# UTRNet-Large architecture/config, mirroring read.py's argparse defaults.
IMG_HEIGHT = 32
IMG_WIDTH = 400
BATCH_MAX_LENGTH = 100
NUM_FIDUCIAL = 20
HIDDEN_SIZE = 256
OUTPUT_CHANNEL = 32  # HRNet override (read.py sets this when FeatureExtraction == "HRNet")
FEATURE_EXTRACTION = "HRNet"
SEQUENCE_MODELING = "DBiLSTM"
PREDICTION = "CTC"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NormalizePAD:
    """Port of dataset.NormalizePAD (vendoring just this avoids dataset.py's
    heavy lmdb/natsort dependencies). Scales to [-1, 1] then right-pads to the
    target width, filling the pad region with the last image column."""

    def __init__(self, max_size, pad_type: str = "right"):
        self.to_tensor = T.ToTensor()
        self.max_size = max_size  # (channels, height, width)
        self.pad_type = pad_type

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.to_tensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.zeros(*self.max_size, dtype=torch.float32)
        pad_img[:, :, :w] = img
        if self.max_size[2] != w:  # border pad
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(
                c, h, self.max_size[2] - w
            )
        return pad_img


class UTRNetModel:
    """Loads the UTRNet checkpoint once and exposes a single `recognize` method."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        charset_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or os.getenv(
            "UTRNET_MODEL_PATH", "saved_models/UTRNet-Large/best_norm_ED.pth"
        )
        self.charset_path = charset_path or os.getenv("UTRNET_CHARSET_PATH", "UrduGlyphs.txt")
        self.device = torch.device(device or os.getenv("UTRNET_DEVICE") or _auto_device())

        # --- Character set + CTC converter (mirrors read.py) -------------- #
        character = self._load_charset(self.charset_path) + " "  # trailing space glyph
        self.converter = CTCLabelConverter(character)
        num_class = len(self.converter.character)  # includes the [CTCblank] at index 0

        # --- Build the architecture and load weights --------------------- #
        opt = SimpleNamespace(
            imgH=IMG_HEIGHT,
            imgW=IMG_WIDTH,
            batch_max_length=BATCH_MAX_LENGTH,
            num_fiducial=NUM_FIDUCIAL,
            input_channel=1,
            output_channel=OUTPUT_CHANNEL,
            hidden_size=HIDDEN_SIZE,
            FeatureExtraction=FEATURE_EXTRACTION,
            SequenceModeling=SEQUENCE_MODELING,
            Prediction=PREDICTION,
            num_class=num_class,
            character=character,
            device=self.device,
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"UTRNet checkpoint not found at '{self.model_path}'. "
                f"Set UTRNET_MODEL_PATH."
            )

        model = Model(opt).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # The checkpoint was saved from a DataParallel-wrapped model, so keys are
        # prefixed with "module.". Strip that to match our bare Model.
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

        self.transform = NormalizePAD((1, IMG_HEIGHT, IMG_WIDTH))

        logger.info(
            "UTRNet loaded (device=%s, num_class=%d, %s/%s/%s)",
            self.device,
            num_class,
            FEATURE_EXTRACTION,
            SEQUENCE_MODELING,
            PREDICTION,
        )

    @staticmethod
    def _load_charset(path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"UTRNet charset not found at '{path}'. Set UTRNET_CHARSET_PATH."
            )
        with open(path, "r", encoding="utf-8") as fh:
            # read.py joins the lines after stripping newlines (no glyph dropped).
            return "".join(line.strip("\n") for line in fh.readlines())

    @torch.inference_mode()
    def recognize(self, image: Image.Image) -> dict:
        """Run UTRNet on a single PIL image and return text + confidence.

        Preprocessing faithfully follows read.py (grayscale, RTL flip,
        aspect-ratio resize, NormalizePAD)."""
        img = image.convert("L")
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Urdu is RTL

        w, h = img.size
        ratio = w / float(h)
        if math.ceil(IMG_HEIGHT * ratio) > IMG_WIDTH:
            resized_w = IMG_WIDTH
        else:
            resized_w = math.ceil(IMG_HEIGHT * ratio)
        img = img.resize((resized_w, IMG_HEIGHT), Image.Resampling.BICUBIC)

        tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1, 1, 32, 400)

        preds = self.model(tensor)  # (1, T, num_class)
        preds_size = torch.IntTensor([preds.size(1)])
        probs = preds.softmax(dim=2)
        max_probs, preds_index = probs.max(dim=2)

        text = self.converter.decode(preds_index, preds_size)[0]
        confidence = self._estimate_confidence(preds_index[0], max_probs[0])

        return {"text": text.strip(), "confidence": confidence, "engine": "utrnet"}

    def recognize_page(self, image: Image.Image) -> dict:
        """Recognize a multi-line page by segmenting it into line crops and
        running UTRNet on each. This is the regime UTRNet was built for; passing
        a whole page straight to `recognize` would squash it to 32px and fail.

        Lines are joined top-to-bottom. Falls back to whole-image recognition if
        segmentation finds nothing (e.g. a single-line crop)."""
        lines = segment_lines(image)
        if not lines:
            result = self.recognize(image)
            result["num_lines"] = 0  # 0 => no segmentation applied
            return result

        texts: List[str] = []
        confidences: List[float] = []
        for _box, crop in lines:
            res = self.recognize(crop)
            if res["text"]:
                texts.append(res["text"])
                confidences.append(res["confidence"])

        return {
            "text": "\n".join(texts),
            "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
            "engine": "utrnet",
            "num_lines": len(lines),
        }

    @staticmethod
    def _estimate_confidence(idx_seq: torch.Tensor, prob_seq: torch.Tensor) -> float:
        """Mean softmax prob over the timesteps the CTC decode actually keeps
        (non-blank, non-repeat) — a best-effort per-image confidence."""
        kept: List[float] = []
        prev = None
        for i, idx in enumerate(idx_seq.tolist()):
            if idx != 0 and not (i > 0 and prev == idx):
                kept.append(float(prob_seq[i]))
            prev = idx
        return sum(kept) / len(kept) if kept else 0.0


# Process-wide singleton so the model is loaded only once.
_MODEL_SINGLETON: Optional[UTRNetModel] = None


def get_model() -> UTRNetModel:
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = UTRNetModel()
    return _MODEL_SINGLETON


def run_utrnet(image: Image.Image) -> dict:
    """Single-line recognition (no segmentation)."""
    return get_model().recognize(image)


def run_utrnet_page(image: Image.Image) -> dict:
    """Page recognition with line segmentation — used by the API layer."""
    return get_model().recognize_page(image)
