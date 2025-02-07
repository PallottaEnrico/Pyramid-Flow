import os
import json
import cv2
import torch
import numpy as np
import PIL
import argparse
from pathlib import Path
from PIL import Image
from einops import rearrange
from video_vae import CausalVideoVAELossWrapper
from torchvision import transforms as pth_transforms
from torchvision.transforms.functional import InterpolationMode
from diffusers.utils import load_image, export_to_video, export_to_gif
from IPython.display import HTML
from eval_utils import SSIM, calculate_ssim
from tqdm import tqdm
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Process video dataset and compute SSIM.")
    parser.add_argument("--src", required=True, type=str, help="Path to source folder containing videos.")
    return parser.parse_args()

def image_transform(images, resize_width, resize_height):
    transform_list = pth_transforms.Compose([
        pth_transforms.Resize((resize_height, resize_width), InterpolationMode.BICUBIC, antialias=True),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return torch.stack([transform_list(image) for image in images])

def get_transform(width, height, new_width=None, new_height=None, resize=False):
    transform_list = []
    if resize:
        if new_width is None:
            new_width = width // 8 * 8
        if new_height is None:
            new_height = height // 8 * 8
        transform_list.append(pth_transforms.Resize((new_height, new_width), InterpolationMode.BICUBIC, antialias=True))
    transform_list.append(pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return pth_transforms.Compose(transform_list)

def load_video_and_transform(video_path, frame_number, new_width=None, new_height=None, max_frames=600, sample_fps=24, resize=False):
    try:
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frames = []
        pil_frames = []
        while True:
            flag, frame = video_capture.read()
            if not flag:
                break
            pil_frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        video_capture.release()
        interval = max(int(fps / sample_fps), 1)
        pil_frames = pil_frames[::interval][:frame_number]
        frames = frames[::interval][:frame_number]
        frames = torch.stack(frames).float() / 255
        width = frames.shape[-1]
        height = frames.shape[-2]
        video_transform = get_transform(width, height, new_width, new_height, resize=resize)
        frames = video_transform(frames)
        pil_frames = [Image.fromarray(frame).convert("RGB") for frame in pil_frames]
        if resize:
            if new_width is None:
                new_width = width // 32 * 32
            if new_height is None:
                new_height = height // 32 * 32
            pil_frames = [frame.resize((new_width, new_height), PIL.Image.BICUBIC) for frame in pil_frames]
        return frames, pil_frames
    except Exception:
        return None

def main():
    args = parse_args()
    train_folder = args.src
    model_path = "pyramid_flow_model/causal_video_vae"
    model_dtype = 'fp32'
    device_id = 0
    torch.cuda.set_device(device_id)
    model = CausalVideoVAELossWrapper(model_path, model_dtype, interpolate=False, add_discriminator=False)
    model = model.to("cuda")
    torch_dtype = torch.float32
    frame_number = 121
    width, height = 700, 480
    original_videos = []
    reconstructed_videos = []
    video_files = sorted(glob(os.path.join(train_folder, "*.mp4")))[:100]
    ssim_model = SSIM()
    to_tensor = pth_transforms.ToTensor()
    for video_path in tqdm(video_files):
        try:
            video_frames_tensor, pil_video_frames = load_video_and_transform(str(video_path), frame_number, new_width=width, new_height=height, resize=False)
            video_frames_tensor = video_frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                latent = model.encode_latent(video_frames_tensor.to("cuda"), sample=False, window_size=8, temporal_chunk=True)
                rec_frames = model.decode_latent(latent.float(), window_size=2, temporal_chunk=True)
            target_size = rec_frames[0].size
            pil_video_frames = [img.resize(target_size, resample=Image.Resampling.BILINEAR) for img in pil_video_frames]
            original_video_tensor = torch.stack([to_tensor(img) for img in pil_video_frames], dim=0)
            rec_video_tensor = torch.stack([to_tensor(img) for img in rec_frames], dim=0)
            original_videos.append(original_video_tensor)
            reconstructed_videos.append(rec_video_tensor)
            export_to_video(pil_video_frames, f'./ori_{video_path.stem}.mp4', fps=24)
            export_to_video(rec_frames, f'./rec_{video_path.stem}.mp4', fps=24)
        except:
            pass
    final_ssim = sum(calculate_ssim(ov.unsqueeze(0), rv.unsqueeze(0), ssim_model) for ov, rv in zip(original_videos, reconstructed_videos) if ov is not None and rv is not None)
    count = len(original_videos)
    result = final_ssim / count if count > 0 else 0
    print(f"Final SSIM: {result}")
    with open("ssim_results.txt", "w") as f:
        f.write(f"Final SSIM: {result}\n")

if __name__ == "__main__":
    main()
