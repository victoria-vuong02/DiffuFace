# SDXL (Stable Diffusion XL) base 1.0 is the core image generation model this entire project builds on.
# It takes a text prompt (and optionally an image) and generates a high-quality 1024x1024 image.
# Everything else — style LoRAs, IP-Adapter-FaceID, inpainting — are add-ons that sit on top of this base.
# You download it once here; it is never retrained, only used as a frozen backbone.

from huggingface_hub import snapshot_download
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

print("Downloading SDXL base 1.0 (~7GB) into models/base/sdxl-base-1.0 ...")

snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="./models/base/sdxl-base-1.0",
    ignore_patterns=["*.bin"],  # prefer .safetensors only
)

print("Download complete. Model is at: models/base/sdxl-base-1.0")
