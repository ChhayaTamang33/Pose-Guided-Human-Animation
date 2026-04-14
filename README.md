# Pose-Guided Human Animation Pipeline

End-to-end pipeline for preprocessing and pose-guided human animation using AnimateAnyone and MimicMotion.

## Overview

This repository provides a complete pipeline for pose-guided human animation, including:

- Video preprocessing and filtering
- Pose extraction using DWPose
- Pose-to-video synthesis using AnimateAnyone
- Alternative inference using MimicMotion
- Evaluation metrics for generated videos

The project is designed for **research reproducibility**, **scalability**, and **modular experimentation**.

## Features

- End-to-end pipeline (raw video → animated output)
- Modular architecture with Hydra-based configuration
- Scalable processing for large datasets (1000+ videos)
- Integration of state-of-the-art models:
  - AnimateAnyone
  - MimicMotion
- Resume capability and logging
- Evaluation metrics (PSNR, SSIM, LPIPS)

## Installation

### 1. Clone repository

git clone https://github.com/<your-username>/pose-guided-human-animation.git
cd pose-guided-human-animation

## Setup

bash setup.sh

---

## 🔷 Usage

### Preprocessing


python -m pgha.pipeline.run_preprocessing

### Pose Extraction (DWpose)

```bash
python -m pgha.pipeline.run_vid2pose_animateAnyone
```
### AnimateAnyone Inference
```bash
python -m pgha.pipeline.run_pose2vid_animateAnyone
```
### MimicMotion Inference
```bash
python -m pgha.pipeline.run_inference_mimicMotion
```
---

## 🔷 Configuration

```markdown
## Configuration

All parameters are controlled via Hydra configs in `configs/`.

Example:

- preprocessing.yaml
- pose2vid_animateAnyone.yaml
- inference_mimicMotion.yaml

You can override parameters:

```bash
python -m pgha.pipeline.run_pose2vid_animateAnyone processing.chunk_size=16
```
---
## 🔷 Notes
- Pretrained weights are downloaded via setup.sh
- External repositories are included as submodules

