#!/bin/bash
#SBATCH --job-name=video_comparison
#SBATCH --partition=RTXA6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

set -e

mkdir -p logs

echo "Job started on $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

COMPARE_DIR=/netscratch/ctamang/dataset/TED
export PYTHONPATH=$COMPARE_DIR:$PYTHONPATH

srun \
    --container-image=/netscratch/ctamang/enroot/moore_train.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER \
    --container-workdir="$COMPARE_DIR" \
    --container-remap-root \
    --export=ALL \
    bash -c '
    echo "Inside container"
    echo "Current directory: $(pwd)"
    
    # Check if lpips is already installed
    python -c "import lpips" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing lpips package..."
        python -m pip install lpips
    else
        echo "lpips already installed"
    fi
    
    # Show installed packages
    echo "Installed packages:"
    python -m pip list | grep -E "lpips|torch|imageio|skimage|pandas"
    
    echo "Running evaluation pipeline..."

    python -m pgha.pipeline.run_evaluation
    
    echo "Script completed with exit code: $?"
'