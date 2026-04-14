#!/bin/bash
set -e

export PYTHONPATH=$(pwd)

echo "Running AnimateAnyone vid2pose..."
python -m pgha.pipeline.run_vid2pose_animateAnyone

echo "Running AnimateAnyone pose2vid..."
python -m pgha.pipeline.run_pose2vid_animateAnyone

echo "Running MimicMotion inference..."
python -m pgha.pipeline.run_inference_mimicMotion

echo "Inference completed."