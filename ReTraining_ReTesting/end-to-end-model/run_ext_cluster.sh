#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=Model
#SBATCH --output=/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/end-to-end-model/cluster_outputs/%j.out
#SBATCH --gres gpu:1

echo "post sbatch commands"

# Credit to Jong Kwon & John McGonigle
abort() { >&2 printf '█%.0s' {1..40}; (>&2 printf "\n[ERROR] $(basename $0) has exited early\n"); exit 1; }
scriptdirpath=$(cd -P -- "$(dirname -- "$0")" && pwd -P);
IFS=$'\n\t'; set -eo pipefail;

trap 'abort' 0; set -u;
pushd "${scriptdirpath}" > /dev/null

echo -e "ml cuda: \n"
ml cuda

# Print hostname and GPU info
echo "Running on node:"
hostname
echo "GPU(s) info:"
nvidia-smi

CONDA_ENV="Mayowa_code_env"

source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "activating ${CONDA_ENV} env"
  set +u; conda activate "${CONDA_ENV}"; set -u
fi

DEBUG=false
if [[ $# -gt 0 && $1 == "--debug" ]]; then
    DEBUG=true
fi

if [ "$DEBUG" == true ]; then
  echo -e "______DEBUG MODE______\n"
  echo -e "python check: \n"
  which python
  
  echo -e "gpu(s) check: \n"
  nvidia-smi

  echo -e "Python, torch, and gpu(s) check \n\n"
  python ~/code/template_scripts/torch/testGPU.py

  echo -e "Checks complete\n"
fi

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

USER="lady6758"
PROJECT_PATH="/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/end-to-end-model"
export LD_LIBRARY_PATH=$(conda info --base)/envs/Mayowa_code_env/lib:$LD_LIBRARY_PATH
export TF_GPU_ALLOCATOR=cuda_malloc_async
python ${PROJECT_PATH}/Model.py

conda deactivate
popd > /dev/null
trap : 0
(>&2 echo "✔")
exit 0
