import os
from typing import Optional

import torch

DEFAULT_THETA = float(os.environ.get("DPP_THETA", -1))
DEFAULT_K = int(os.environ.get("DPP_K", 8))


class DppTimeSelector(object):
    """Legacy selector kept for compatibility; ranks by query-frame similarity."""

    def __init__(self, k_selection: int = DEFAULT_K, theta: float = DEFAULT_THETA, clip_service_url: str = None):
        super().__init__()
        self.k_selection = k_selection
        self.theta = theta
        self.clip_service_url = clip_service_url

    def __call__(self, frames, prompts, frame_embeddings: Optional[torch.Tensor] = None):
        if frame_embeddings is None:
            raise ValueError("frame_embeddings is required in compatibility mode.")
        n = frame_embeddings.shape[0]
        if n <= self.k_selection:
            return list(range(n))
        scores = frame_embeddings.mean(dim=1)
        return sorted(torch.topk(scores, k=self.k_selection).indices.tolist())
