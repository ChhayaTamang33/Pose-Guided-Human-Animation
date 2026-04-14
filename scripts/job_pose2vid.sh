#!/bin/bash
#SBATCH --job-name=mimicInference
#SBATCH --partition=RTXA6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

mkdir -p logs

echo "Job started on $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

REPO_DIR=/netscratch/ctamang/dataset/TED
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

srun \
    --container-image=/netscratch/ctamang/enroot/moore_train.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER \
    --container-workdir="$REPO_DIR" \
    --export=ALL \
    bash -c '

echo "Inside container"

# Fix CUDA library mismatch (ONNXRuntime expects CUDA11 names)
CUDA_LIB=/usr/local/cuda-12.4/targets/x86_64-linux/lib

if [ -f $CUDA_LIB/libcublasLt.so.12 ] && [ ! -f $CUDA_LIB/libcublasLt.so.11 ]; then
    echo "Creating CUDA compatibility symlink"
    ln -s $CUDA_LIB/libcublasLt.so.12 $CUDA_LIB/libcublasLt.so.11
fi

export LD_LIBRARY_PATH=$CUDA_LIB:$LD_LIBRARY_PATH

echo "CUDA libraries available:"
ls $CUDA_LIB | grep cublas || true

echo "ONNX Runtime providers:"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

echo "Installing correct diffusers version"
pip uninstall diffusers -y || true
pip install diffusers==0.27.0

echo "Running mimicMotion inference..."
python -m pgha.pipeline.run_inference_mimicMotion
'