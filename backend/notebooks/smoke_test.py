# Verifies that SDXL loads correctly and MPS (Apple Silicon GPU) is working.
# Run this once after downloading the model. If it produces test_output.png, Phase 0 is complete.

import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path

print("=== DiffuFace Smoke Test ===\n")

# Check MPS
print(f"PyTorch version : {torch.__version__}")
print(f"MPS available   : {torch.backends.mps.is_available()}")
print(f"MPS built       : {torch.backends.mps.is_built()}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device    : {device}\n")

# Load pipeline
model_path = Path(__file__).parent.parent / "models/base/sdxl-base-1.0"
print(f"Loading model from: {model_path}")
print("(This takes 1-2 minutes on first load while weights move into memory...)\n")

pipe = StableDiffusionXLPipeline.from_pretrained(
    str(model_path),
    torch_dtype=torch.float32,  # float16 produces NaN latents on MPS, causing black images
    use_safetensors=True,
    variant="fp16",  # loads fp16 disk files, upcasts to float32 in memory — same result, smaller files
)
pipe = pipe.to(device)

# Reduce memory usage — slices attention and VAE decoding into smaller chunks
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("Model loaded. Running generation...\n")

image = pipe(
    prompt="a red apple on a white table, studio lighting, photorealistic",
    num_inference_steps=20,
    guidance_scale=7.5,
).images[0]

output_path = Path(__file__).parent.parent / "test_output.png"
image.save(str(output_path))

print(f"Done. Image saved to: {output_path}")
print("\nPhase 0 complete — SDXL is working on your machine.")
