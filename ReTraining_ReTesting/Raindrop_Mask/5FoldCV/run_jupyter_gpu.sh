#!/bin/bash
#SBATCH --time=2:00:00              # 2 hours of runtime
#SBATCH --job-name=jupyter_gpu
#SBATCH --output=/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/GPUClusterOutputs/%j.out
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=16G                  # Memory allocation (e.g., 16GB)

# Load modules if needed (e.g. CUDA)
module load cuda

# Activate your conda environment
CONDA_ENV="Mayowa_code_env"
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Optionally, check GPU visibility
nvidia-smi

# Launch Jupyter on a specific port (here: 8888) without opening a browser
# Use --no-browser so it doesn't try to open a browser on the cluster
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Note: The script will stay "running" while Jupyter is open
# When you exit the notebook (Ctrl-C in the running job),
# the script ends, and SLURM job is released.
