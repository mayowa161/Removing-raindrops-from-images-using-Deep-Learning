#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=ValAttentionMask
#SBATCH --output=/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/cluster_outputs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2


echo "post sbatch commands"

# Credit to Jong Kwon & John McGonigle
abort() { >&2 printf '█%.0s' {1..40}; (>&2 printf "\n[ERROR] $(basename $0) has exited early\n"); exit 1; }  # print error message
scriptdirpath=$(cd -P -- "$(dirname -- "$0")" && pwd -P);
IFS=$'\n\t'; set -eo pipefail; # exits if error, and set IFS, so no whitespace error

trap 'abort' 0; set -u;
# Sets abort trap defined in line 2, set -u exits when detects unset variables

# cd into the scriptdirpath so that relative paths work
pushd "${scriptdirpath}" > /dev/null

# _________ ACTUAL CODE THAT RUNS STUFF __________


echo -e "ml cuda: \n"
ml cuda

CONDA_ENV="Mayowa_code_env"

# Activate conda env if in base env, or don't if already set.
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "activating ${CONDA_ENV} env"
  set +u; conda activate "${CONDA_ENV}"; set -u
fi

# Perform checks if debug is true
DEBUG=false
# Check if the debug argument is provided
if [[ $# -gt 0 && $1 == "--debug" ]]; then
    DEBUG=true
fi

if [ "$DEBUG" == true ]; then
  # Check python being used
  echo -e "______DEBUG MODE______\n"
  echo -e "python check: \n"
  which python
  
  echo -e "gpu(s) check: \n"
  nvidia-smi

  echo -e "Python, torch, and gpu(s) check \n\n"
  python ~/code/template_scripts/torch/testGPU.py

  echo -e "Checks complete\n"
fi

USER="lady6758"
PROJECT_PATH="/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask"
# Run the actual script
export LD_LIBRARY_PATH=$(conda info --base)/envs/Mayowa_code_env/lib:$LD_LIBRARY_PATH
export TF_GPU_ALLOCATOR=cuda_malloc_async
python ${PROJECT_PATH}/ValAttentionMaskRealSyn.py

      


conda deactivate

# ___________ MORE SAFE CRASH JARGON ____________

popd > /dev/null

trap : 0
(>&2 echo "✔")
exit 0