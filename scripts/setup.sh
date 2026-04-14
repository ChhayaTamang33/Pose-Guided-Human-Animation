# scripts/setup.sh

#!/bin/bash
set -e

echo "Setting up submodules..."

# If external dirs don't exist or are empty, clone with explicit names
if [ ! -d "external/animateanyone/.git" ]; then
    git clone https://github.com/MooreThreads/Moore-AnimateAnyone.git external/animateanyone
fi

if [ ! -d "external/mimicmotion/.git" ]; then
    git clone https://github.com/Tencent/MimicMotion.git external/mimicmotion
fi

echo ""
echo "Downloading DWPose weights..."
DWPOSE_DIR="external/animateanyone/pretrained_weights/DWPose"
mkdir -p $DWPOSE_DIR

# Download if not present
if [ ! -f "$DWPOSE_DIR/yolox_l.onnx" ]; then
    wget -P $DWPOSE_DIR https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx
fi
if [ ! -f "$DWPOSE_DIR/dw-ll_ucoco_384.onnx" ]; then
    wget -P $DWPOSE_DIR https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx
fi

echo "Setup complete."