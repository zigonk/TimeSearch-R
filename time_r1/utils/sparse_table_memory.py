import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class BeamCandidate:
    score: float
    start: int
    end: int


class SparseTableMemory:
    """
    Sparse table memory over per-frame embeddings.

    Entry definition:
      T[i][j] summarizes [i - 2^j + 1, i]
    """

    def __init__(self, max_frames: int = 4096):
        self.max_frames = max_frames
        self.max_level = int(math.log2(max_frames)) + 1
        self.table: Dict[Tuple[int, int], torch.Tensor] = {}
        self.current_frame = -1
        self.hidden_size: Optional[int] = None

    def reset(self):
        self.table.clear()
        self.current_frame = -1
        self.hidden_size = None

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self.table

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1)

    def aggregate(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        # Mean aggregation in embedding space, then normalize.
        return self._normalize((emb1 + emb2) * 0.5)

    def update(self, frame_embedding: torch.Tensor):
        """Append one frame embedding and update O(log N) sparse table entries."""
        if frame_embedding.dim() != 1:
            frame_embedding = frame_embedding.view(-1)
        frame_embedding = self._normalize(frame_embedding)

        self.current_frame += 1
        i = self.current_frame
        self.hidden_size = frame_embedding.shape[0]
        self.table[(i, 0)] = frame_embedding.clone()

        j = 1
        while i >= (1 << j) - 1:
            left_end = i - (1 << (j - 1))
            left_key = (left_end, j - 1)
            right_key = (i, j - 1)
            if left_key in self and right_key in self:
                self.table[(i, j)] = self.aggregate(self.table[left_key], self.table[right_key])
            else:
                break
            j += 1

    def build_from_embeddings(self, embeddings: torch.Tensor):
        """Build sparse table from a [num_frames, hidden] embedding tensor."""
        self.reset()
        for emb in embeddings:
            self.update(emb)

    def _entry_for_segment(self, start: int, end: int) -> Tuple[int, int, int, int]:
        """
        Return sparse-table entry that best fits [start, end] using power-of-two length.
        Output: (entry_start, entry_end, entry_level, entry_end_idx)
        """
        length = end - start + 1
        level = int(math.floor(math.log2(length)))
        span = 1 << level
        entry_end = start + span - 1
        return start, entry_end, level, entry_end

    def _score_segment(self, query_embedding: torch.Tensor, start: int, end: int) -> float:
        entry_start, entry_end, level, entry_end_idx = self._entry_for_segment(start, end)
        key = (entry_end_idx, level)
        if key not in self:
            return -1e9
        q = self._normalize(query_embedding.view(-1))
        v = self.table[key]
        return float((q * v).sum().item())

    @staticmethod
    def _split_segment(start: int, end: int) -> List[Tuple[int, int]]:
        """
        Split [start, end] into two overlapping half-like segments:
        [start, start + 2^k - 1] and [end - 2^k + 1, end], where 2^k ~= half length.
        """
        length = end - start + 1
        if length <= 1:
            return []
        half = max(1, length // 2)
        k = int(round(math.log2(half)))
        span = max(1, 1 << k)
        left = (start, min(end, start + span - 1))
        right = (max(start, end - span + 1), end)
        if left == right:
            return [left]
        return [left, right]

    def beam_search_frames(
        self,
        query_embedding: torch.Tensor,
        start: int,
        end: int,
        top_k: int,
        beam_width: int = 4,
    ) -> List[int]:
        """
        Query-guided beam search from [start, end] down to leaf frames.
        """
        if self.current_frame < 0:
            return []

        start = max(0, start)
        end = min(end, self.current_frame)
        if start > end:
            return []

        beams: List[BeamCandidate] = [BeamCandidate(self._score_segment(query_embedding, start, end), start, end)]
        leaf_scores: Dict[int, float] = {}

        # Conservative max steps to ensure termination.
        max_steps = int(math.ceil(math.log2(max(1, end - start + 1)))) + 3

        for _ in range(max_steps):
            expanded: List[BeamCandidate] = []
            all_leaf = True

            for cand in beams:
                if cand.start == cand.end:
                    prev = leaf_scores.get(cand.start, -1e9)
                    if cand.score > prev:
                        leaf_scores[cand.start] = cand.score
                    continue

                all_leaf = False
                for child_start, child_end in self._split_segment(cand.start, cand.end):
                    score = self._score_segment(query_embedding, child_start, child_end)
                    expanded.append(BeamCandidate(score, child_start, child_end))

            if all_leaf:
                break
            if not expanded:
                break

            expanded.sort(key=lambda x: x.score, reverse=True)
            beams = expanded[:beam_width]

        # Safety: add any remaining beam leaves.
        for cand in beams:
            if cand.start == cand.end:
                prev = leaf_scores.get(cand.start, -1e9)
                if cand.score > prev:
                    leaf_scores[cand.start] = cand.score

        if not leaf_scores:
            # fallback uniform within [start, end]
            if top_k >= (end - start + 1):
                return list(range(start, end + 1))
            return (
                torch.linspace(start, end, steps=top_k)
                .round()
                .long()
                .tolist()
            )

        ranked = sorted(leaf_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in ranked[:top_k]]
        selected.sort()

        if len(selected) < top_k:
            pool = list(range(start, end + 1))
            for idx in pool:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= top_k:
                    break
            selected.sort()

        return selected[:top_k]
