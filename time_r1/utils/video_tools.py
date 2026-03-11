import math
import mimetypes
import os
from typing import Dict, List, Union

import numpy as np
import torch
from qwen_agent.tools.base import BaseTool, TOOL_REGISTRY, register_tool

from time_r1.prompt import get_prompt_fn
from time_r1.utils.clip_service import SiglipClient
from time_r1.utils.sparse_table_memory import SparseTableMemory

MAX_NUM_KEY_FRAMES = int(os.environ.get("MAX_NUM_KEY_FRAMES", 8))
FORCE_NUM_KEY_FRAMES = int(os.environ.get("FORCE_NUM_KEY_FRAMES", 0))
FORCE_UNIFORM_SAMPLING = int(os.environ.get("FORCE_UNIFORM_SAMPLING", 0))
MEMORY_BEAM_WIDTH = int(os.environ.get("MEMORY_BEAM_WIDTH", 4))
PROMPT_FN = get_prompt_fn("tool_response")

text_encoder = SiglipClient()


def construct_temporal_augmented_frames(timestamps: List[float], frames: torch.Tensor):
    content = []
    for t, frame in zip(timestamps, frames):
        content.append({"type": "text", "text": f"{t:.1f}s"})
        content.append({"type": "image", "image": frame})
    return content


@register_tool("seek_video_frames", allow_overwrite=True)
class VideoFrameSeeker(BaseTool):
    description = (
        "Search and select video frames according to textual query and temporal window. "
        "Time is in seconds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query describes objects/scenes/events of interest.",
            },
            "start_time": {"type": "number", "description": "Start time in seconds."},
            "end_time": {"type": "number", "description": "End time in seconds."},
            "num_frames": {
                "type": "integer",
                "description": f"Number of frames to sample (maximum {MAX_NUM_KEY_FRAMES}).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def cast(self, params):
        if "start_time" in params and params["start_time"] is not None:
            try:
                params["start_time"] = float(params["start_time"])
            except Exception:
                params["start_time"] = 0
        else:
            params["start_time"] = 0

        if "end_time" in params and params["end_time"] is not None:
            try:
                params["end_time"] = float(params["end_time"])
            except Exception:
                params["end_time"] = 0xFFFF
        else:
            params["end_time"] = 0xFFFF

        if "num_frames" in params and params["num_frames"] is not None:
            try:
                params["num_frames"] = int(params["num_frames"])
            except Exception:
                params["num_frames"] = MAX_NUM_KEY_FRAMES
        else:
            params["num_frames"] = MAX_NUM_KEY_FRAMES

        params["num_frames"] = max(1, min(params["num_frames"], MAX_NUM_KEY_FRAMES))

        if "query" in params and params["query"] is not None:
            params["query"] = str(params["query"])
            if mimetypes.guess_type(params["query"])[0]:
                params["query"] = params["query"] + "  \t."
        else:
            params["query"] = ""
        return params

    def call(self, params: Union[str, dict], multimodal_cache: Dict, **kwargs):
        params = self.cast(params)
        params = self._verify_json_format_args(params)
        return self.seek_video_frames(
            params["query"],
            params["start_time"],
            params["end_time"],
            params["num_frames"],
            multimodal_cache=multimodal_cache,
        )

    @staticmethod
    def _uniform_indices(start_idx: int, end_idx: int, num_frames: int) -> List[int]:
        length = end_idx - start_idx + 1
        if length <= num_frames:
            return list(range(start_idx, end_idx + 1))
        return (
            torch.linspace(start_idx, end_idx, steps=num_frames)
            .round()
            .long()
            .tolist()
        )

    def seek_video_frames(self, query: str, start_time: int, end_time: int, num_frames: int, multimodal_cache: Dict):
        fps = multimodal_cache["fps"]
        frames = multimodal_cache["video"]

        start_frame_idx = max(math.floor(start_time * fps), 0)
        end_frame_idx = min(math.ceil(end_time * fps), len(frames) - 1)

        if end_frame_idx < start_frame_idx:
            end_frame_idx = start_frame_idx

        num_frames = max(min(num_frames, end_frame_idx - start_frame_idx + 1, MAX_NUM_KEY_FRAMES), 1)
        if FORCE_NUM_KEY_FRAMES > 0:
            num_frames = min(FORCE_NUM_KEY_FRAMES, end_frame_idx - start_frame_idx + 1)

        if query and not FORCE_UNIFORM_SAMPLING:
            memory = multimodal_cache.get("sparse_memory")
            if memory is None:
                embeddings = multimodal_cache["embedding"]
                memory = SparseTableMemory(max_frames=max(len(embeddings), 2))
                memory.build_from_embeddings(embeddings)
                multimodal_cache["sparse_memory"] = memory

            query_embedding = text_encoder.encode_texts([query]).squeeze(0)
            global_idx = memory.beam_search_frames(
                query_embedding=query_embedding,
                start=start_frame_idx,
                end=end_frame_idx,
                top_k=num_frames,
                beam_width=MEMORY_BEAM_WIDTH,
            )
            if len(global_idx) == 0:
                global_idx = self._uniform_indices(start_frame_idx, end_frame_idx, num_frames)
        else:
            global_idx = self._uniform_indices(start_frame_idx, end_frame_idx, num_frames)

        local_idx = [i - start_frame_idx for i in global_idx]
        frames_to_select = frames[start_frame_idx : end_frame_idx + 1]
        local_idx_tensor = torch.tensor(local_idx, device=frames_to_select.device)
        local_idx_tensor = torch.clamp(local_idx_tensor, 0, len(frames_to_select) - 1)
        selected_frames = frames_to_select.index_select(0, local_idx_tensor)

        timestamps = [gid / fps for gid in global_idx]
        question = multimodal_cache["question"]
        duration = multimodal_cache["duration"]
        response_content = construct_temporal_augmented_frames(timestamps, selected_frames)
        timestamps_str = ",".join([f"{t:.1f}s" for t in timestamps])
        response_content.append(
            {
                "type": "text",
                "text": PROMPT_FN({"timestamps": timestamps_str, "question": question, "duration": duration}),
            }
        )
        return response_content


def get_video_tool_by_name(fn_name: str):
    tool_cls = TOOL_REGISTRY.get(fn_name)
    if tool_cls is None:
        raise ValueError(f"Unknown function name: {fn_name}")
    return tool_cls()


def video_tool_call(params: Dict, multimodal_cache: Dict):
    func = params.get("function", {})
    fn_name = func.get("name", "unknown")
    fn_args = func.get("arguments", {})
    try:
        tool_response = get_video_tool_by_name(fn_name).call(fn_args, multimodal_cache)
        return {
            "role": "tool",
            "name": fn_name,
            "arguments": fn_args,
            "content": tool_response,
        }
    except Exception as e:
        print(f"Failed to call tool function: {fn_name=}, {fn_args=}, got err {e}, duration: {multimodal_cache.get('duration', 0)}")
        return None


if __name__ == "__main__":
    from time_r1.utils.qwen_vl_utils import fetch_video

    frames = fetch_video({"video": "workdir/datasets/Charades/videos/0A8CF.mp4"})
    print(frames.shape)

    mm_cache = {
        "video": frames,
        "embedding": torch.randn(frames.shape[0], 1024),
        "fps": 1,
        "question": "What is the man doing?",
        "duration": 100,
    }
    mem = SparseTableMemory(max_frames=max(len(mm_cache["embedding"]), 2))
    mem.build_from_embeddings(mm_cache["embedding"])
    mm_cache["sparse_memory"] = mem

    resp = video_tool_call(
        {
            "function": {
                "name": "seek_video_frames",
                "arguments": {
                    "query": "man",
                },
            },
        },
        mm_cache,
    )
    print(resp)
