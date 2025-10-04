# CLIP-Based Video Semantic Scoring

## Overview
This project uses **OpenCLIP** to analyze videos and measure how well their visual content matches one or more **text prompts** (e.g., *"there are mirrors"*, *"a person walking"*, *"a white car on the street"*).

It automatically scans a specified directory for video files, samples frames from each video, and computes **CLIP similarity scores** between the frames and the given prompts.

For each video, the tool reports:
- **Maximum similarity** across sampled frames  
- **Average similarity** score  
- **Per-frame** similarity values for deeper inspection

---

## Features
- Supports both **local CLIP weights** and **internet-downloaded** pretrained models  
- Works with multiple prompts at once  
- Automatically discovers all video files under a given directory  
- CUDA support (uses GPU if available)  
- Detailed per-video output (max, average, per-frame)

---

## Example Usage

### Using Local Weights
```bash
python Query_Videos_OpenClip.py --root "D:\Videos" --clip-dir "D:\Pretrained\Model\open_clip_model.safetensors" --text "people are playing football"
