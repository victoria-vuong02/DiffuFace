# DiffuFace — Face Generator

<table>
<tr>
<td width="40%" valign="top">
<img width="572" height="1024" alt="Main_img" src="https://github.com/user-attachments/assets/f8e4b429-60d0-4e85-b221-7694efec7285" />
</td>
<td valign="top">

### What it does

Upload a face photo. DiffuFace does two things:

**1. Stylize it** — turn it into anime, watercolor, claymation, pixel art, or oil painting while keeping it recognizably *you* (same face shape, expression, pose)

**2. Edit a specific region** — draw over any area (hair, background, clothing), type what you want there, and only that region changes — everything else stays untouched

Built on SDXL with custom-trained style LoRAs + IP-Adapter-FaceID for identity preservation.

</td>
</tr>
</table>

---

## Setup

### Backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Download base model
```bash
cd backend
python notebooks/download_sdxl.py
```

### Verify environment
```bash
cd backend
python notebooks/smoke_test.py
# Generates test_output.png — confirms SDXL is running on Apple Silicon (MPS)
```

---

## Training Pipeline

```
Raw style images (20–25 per style)
        ↓
Resize to 1024×1024  →  preprocessing/resize_and_crop.py
        ↓
Auto-caption + manual review  →  preprocessing/auto_caption.py
        ↓
Deduplicate  →  preprocessing/dedup_filter.py
        ↓
LoRA training (kohya_ss, one per style)  →  training/train_lora.sh
        ↓
models/loras/<style>.safetensors
        ↓
Face alignment  →  preprocessing/face_align.py
        ↓
Identity embedding (insightface)
        ↓
SDXL + LoRA + IP-Adapter-FaceID  →  inference/stylize.py
        ↓
Stylized output (identity preserved)
```

---

## Tech Stack

- **Python** — primary language
- **PyTorch** (MPS backend) — runs all models on Apple Silicon GPU
- **diffusers** (Hugging Face) — SDXL pipeline, LoRA loading, inpainting
- **SDXL base 1.0** — core image generation backbone
- **kohya_ss / sd-scripts** — LoRA training for each art style
- **IP-Adapter-FaceID** — face identity conditioning (keeps output looking like the input person)
- **insightface** — face detection and identity embedding extraction
- **ControlNet** *(optional)* — pose/angle consistency across styles

---

## Project Structure

```
main/
├── frontend/
│
└── backend/
    ├── requirements.txt
    │
    ├── data/
    │   ├── raw_styles/                # collected/generated images, one folder per style
    │   │   ├── anime/
    │   │   ├── watercolor/
    │   │   ├── claymation/
    │   │   ├── pixel_art/
    │   │   └── oil_painting/
    │   ├── processed_styles/          # after resize, caption, dedup
    │   │   └── <style>/
    │   │       ├── images/
    │   │       └── captions/
    │   └── test_faces/
    │
    ├── preprocessing/
    │   ├── resize_and_crop.py         # standardize all training images to 1024×1024
    │   ├── auto_caption.py            # BLIP/WD14 tagger to generate starter captions
    │   ├── face_align.py              # detect + align faces before inference
    │   └── dedup_filter.py            # remove near-duplicate images (prevents overfitting)
    │
    ├── training/
    │   ├── configs/                   # one kohya_ss .toml config per style
    │   ├── train_lora.sh              # wrapper script that calls sd-scripts with a config
    │   └── training_logs/
    │
    ├── models/
    │   ├── base/                      # SDXL base 1.0 (downloaded, not in git)
    │   ├── loras/                     # trained .safetensors LoRA files, one per style
    │   ├── ip_adapter/                # IP-Adapter-FaceID weights (downloaded)
    │   └── controlnet/
    │
    ├── inference/
    │   ├── stylize.py                 # photo + style name → stylized output
    │   ├── inpaint.py                 # photo + mask + prompt → region-edited output
    │   ├── pipeline_utils.py          # shared model loading and setup
    │   └── mask_utils.py              # converts drawn region into inpainting mask
    │
    ├── evaluation/
    │   ├── identity_similarity.py     # cosine similarity between input/output face embeddings
    │   ├── style_consistency_check.py # CLIP-based style fidelity scoring
    │   ├── compare_styles_grid.py     # side-by-side grid of all 5 styles for one face
    │   └── eval_results/
    │
    └── notebooks/
        ├── download_sdxl.py           # one-time model download script
        └── smoke_test.py              # verify SDXL + MPS is working
```

---

## Features

- [x] Environment setup — SDXL running on Apple Silicon (MPS, float32)
- [ ] Style data collection — 20–25 images per style (anime, watercolor, claymation, pixel art, oil painting)
- [ ] Preprocessing pipeline — resize, caption, dedup
- [ ] LoRA training — 5 custom style adapters via kohya_ss
- [ ] Identity-preserving stylization — SDXL + LoRA + IP-Adapter-FaceID
- [ ] Prompted region editing — mask-based inpainting with SDXL
- [ ] Quantitative evaluation — face embedding similarity + style consistency scoring
- [ ] Web UI + REST API deployment

---

## Future ML Upgrades

Planned improvements that replace pretrained components with self-trained equivalents:

- [ ] **Face identity encoder** — train ResNet-50 with ArcFace loss on CelebA (replaces InsightFace)
- [ ] **Style quality classifier** — train EfficientNet-B3 on WikiArt (replaces CLIP similarity scoring)
- [ ] **Mask refinement U-Net** — train to remove inpainting seam artifacts
- [ ] **Custom IP-Adapter** — implement and train from scratch following the original paper

---

## Datasets

| Dataset | Used for |
|---|---|
| WikiArt (~80k artworks) | Style LoRA training |
| CelebA (~200k face images) | Future: face identity encoder training |
| Synthetic SDXL-generated sets | Style LoRA training |
| Own face photos | Inference testing and evaluation |
