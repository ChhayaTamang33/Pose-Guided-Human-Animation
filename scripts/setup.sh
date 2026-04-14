#!/bin/bash
set -e

echo "Setting up submodules..."

git submodule update --init --recursive

echo "Submodules ready."

echo ""
echo "Installing Python dependencies..."

pip install --upgrade pip
pip install -r requirements.txt

echo "Dependencies installed."

echo ""

echo ""
echo "Downloading DWPose weights..."

DWPOSE_DIR="external/animateanyone/pretrained_weights/DWPose"
mkdir -p "$DWPOSE_DIR"

# yolox_l.onnx
if [ ! -f "$DWPOSE_DIR/yolox_l.onnx" ]; then
    echo "Downloading yolox_l.onnx..."
    wget -q --show-progress -O "$DWPOSE_DIR/yolox_l.onnx" \
    https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx
else
    echo "yolox_l.onnx already exists."
fi

# dw-ll_ucoco_384.onnx
if [ ! -f "$DWPOSE_DIR/dw-ll_ucoco_384.onnx" ]; then
    echo "Downloading dw-ll_ucoco_384.onnx..."
    wget -q --show-progress -O "$DWPOSE_DIR/dw-ll_ucoco_384.onnx" \
    https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx
else
    echo "dw-ll_ucoco_384.onnx already exists."
fi

echo "DWPose weights ready (AnimateAnyone)."

echo ""
echo "[4/4] Downloading MimicMotion weights..."

MIMIC_DIR="external/mimicmotion/models"
DWPOSE_MIMIC_DIR="$MIMIC_DIR/DWPose"

mkdir -p "$DWPOSE_MIMIC_DIR"

# DWPose weights for MimicMotion
if [ ! -f "$DWPOSE_MIMIC_DIR/yolox_l.onnx" ]; then
    echo "Downloading MimicMotion yolox_l.onnx..."
    wget -q --show-progress \
        https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx \
        -O "$DWPOSE_MIMIC_DIR/yolox_l.onnx"
else
    echo "MimicMotion yolox_l.onnx already exists."
fi

if [ ! -f "$DWPOSE_MIMIC_DIR/dw-ll_ucoco_384.onnx" ]; then
    echo "Downloading MimicMotion dw-ll_ucoco_384.onnx..."
    wget -q --show-progress \
        https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx \
        -O "$DWPOSE_MIMIC_DIR/dw-ll_ucoco_384.onnx"
else
    echo "MimicMotion dw-ll_ucoco_384.onnx already exists."
fi

# MimicMotion model
if [ ! -f "$MIMIC_DIR/MimicMotion_1-1.pth" ]; then
    echo "Downloading MimicMotion model..."
    wget -q --show-progress \
        https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth \
        -O "$MIMIC_DIR/MimicMotion_1-1.pth"
else
    echo "MimicMotion model already exists."
fi

echo "MimicMotion weights ready."

echo "Setup complete."