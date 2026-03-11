import glob
import multiprocessing
import os

import torch
import tqdm

from time_r1.utils.clip_service import SiglipClient
from time_r1.utils.io import load_jsonl


SIGLIP_URL = os.environ.get("SIGLIP_URL", "grpc://127.0.0.1:51000")
clip_model = SiglipClient(base_url=SIGLIP_URL)

def process_single_video(video_path):
    try:
        video = torch.load(video_path + ".frame_cache")["frame_tensor"]
        features = clip_model.encode_images(video)
        print(features.shape, video.shape)
        torch.save(features, video_path + ".feature_cache")
    except Exception as e:
        print(f"{e}, {video_path}")


def prepare_feature_cache(video_root, dataset_path=None, num_workers=8, overwrite=False):
    if dataset_path is not None:
        video_list = load_jsonl(dataset_path)
        video_list = [os.path.join(video_root, v["video"]) for v in video_list]
    else:
        video_list = glob.glob(os.path.join(video_root, "*.mp4"))

    if not video_list:
        print(f"No MP4 videos found in {video_root}")
        return
    if not overwrite:
        print("skipping videos that already have feature cache")
        num_total = len(video_list)
        video_list = [v for v in video_list if not os.path.exists(v + ".feature_cache")]
        num_skipped = num_total - len(video_list)
        print(f"skipped {num_skipped} videos")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Found {len(video_list)} videos. Starting processing with {num_workers} workers...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_single_video, video_list), total=len(video_list)))

    print("All videos processed.")


if __name__ == "__main__":
    import fire

    fire.Fire(prepare_feature_cache)
