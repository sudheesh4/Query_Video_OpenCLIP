import argparse
import copy
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import open_clip
from typing import List, Dict, Tuple


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

@torch.no_grad()
def prompt_decision(
    frame_img: Image.Image,
    clip_model,
    clip_preprocess,
    text_emb: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Returns similarity scores (one per prompt) for a single frame.

    frame_img: PIL.Image
    text_emb: [num_prompts, D] normalized text embeddings
    """
    img_tensor = clip_preprocess(frame_img).unsqueeze(0).to(device)
    v = clip_model.encode_image(img_tensor)
    v = v / v.norm(dim=-1, keepdim=True)  # [1, D]
    sims = (v @ text_emb.T).squeeze(0)    # [num_prompts]
    return sims  # higher is more similar

def sample_frames(video_path: str, every_n: int = 12, max_frames: int = 8) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames

def find_videos(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]

from pathlib import Path

def load_model(model_name: str, clip_dir):
    if clip_dir:
        p = Path(clip_dir)
        if p.is_dir():
            # find a weights file inside the folder
            cand = sorted(
                [q for q in p.glob("*") if q.suffix.lower() in {".pt", ".bin", ".safetensors"}]
            )
            if not cand:
                raise FileNotFoundError(f"No CLIP weights found in: {p}")
            if len(cand) > 1:
                print(f"Warning: multiple weight files found, using: {cand[0].name}")
            weights_path = str(cand[0])
        else:
            weights_path = str(p)

        clip_model, _, clip_pre = open_clip.create_model_and_transforms(
            model_name, pretrained=weights_path
        )
    else:
        # downloads from the internet
        clip_model, _, clip_pre = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
    clip_tok = open_clip.get_tokenizer(model_name)
    return clip_model, clip_pre, clip_tok

def compute_text_embedding(clip_model, clip_tok, prompts: List[str], device: str) -> torch.Tensor:
    toks = clip_tok(prompts).to(device)
    with torch.no_grad():
        t = clip_model.encode_text(toks)
        t = t / t.norm(dim=-1, keepdim=True)
    return t  # [num_prompts, D]

def score_videos(
    videos: List[Path],
    clip_model,
    clip_pre,
    text_emb: torch.Tensor,
    device: str,
    every_n: int,
    max_frames: int
) -> Dict[str, Tuple[float, float, List[float]]]:
    """
    For each video, sample frames and compute similarity.
    If multiple prompts are provided, we take the max similarity across prompts for each frame.
    Returns: dict[video_path] = (max_over_frames, avg_over_frames, per_frame_scores)
    """
    results = {}
    for v in videos:
        frames = sample_frames(str(v), every_n=every_n, max_frames=max_frames)
        per_frame = []
        for f in frames:
            sims = prompt_decision(f, clip_model, clip_pre, text_emb, device)  # [num_prompts]
            per_frame.append(float(sims.max().item()))
        if per_frame:
            mx = max(per_frame)
            avg = float(np.mean(np.array(per_frame)))
            results[str(v)] = copy.deepcopy((mx, avg, per_frame))
            print(f">>>>> {v}")
        else:
            results[str(v)] = (float("nan"), float("nan"), [])
            print(f">>>>> {v} (no frames sampled)")
    return results

def main():
    parser = argparse.ArgumentParser(description="CLIP-based video prompt scoring")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to recursively search for videos"
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        required=True,
        help="One or more text prompts (space-separated). For phrases, quote them."
    )
    parser.add_argument(
        "--clip-dir",
        type=str,
        default=None,
        help="Local path to CLIP weights (e.g., .pt/.safetensors). If not provided, downloads from the internet."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="open_clip model name (default: ViT-B-32)"
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=12,
        help="Sample every N frames (default: 12)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Maximum frames to sample per video (default: 8)"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"--root must be an existing directory. Got: {root}")

    prompts = args.text
    if not prompts:
        raise ValueError("At least one --text prompt is required.")

    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")
    print(f"Using local weights: {bool(args.clip_dir)}")

    # Load model (local if provided, otherwise from internet)
    clip_model, clip_pre, clip_tok = load_model(args.model, args.clip_dir)
    clip_model = clip_model.to(DEVICE).eval()

    # Encode prompts
    text_emb = compute_text_embedding(clip_model, clip_tok, prompts, DEVICE)

    # Collect videos
    videos = find_videos(root)
    if not videos:
        print("No videos found under the given --root.")
        return

    # Score videos
    results = score_videos(
        videos=videos,
        clip_model=clip_model,
        clip_pre=clip_pre,
        text_emb=text_emb,
        device=DEVICE,
        every_n=args.every_n,
        max_frames=args.max_frames
    )

    # Print summary
    print("\n>>>>>> Prompts:")
    for p in prompts:
        print(f" - {p}")

    for vid, (mx, avg, per_frame) in results.items():
        print(f"\n >>>>>> Video : {vid}")
        print(f"\t Max : {mx}, Avg : {avg}")
        print(f"\n Frames: {per_frame}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>")

if __name__ == "__main__":
    main()
