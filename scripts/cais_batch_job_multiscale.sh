#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=high_priority
#SBATCH --job-name=fineweb_edu

# Setup distributed args
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "Batch script head node IP: ${MASTER_ADDR} / # nodes: ${SLURM_JOB_NUM_NODES}"

srun bash -c "scripts/full_train.sh \
        ${MASTER_ADDR} ${SLURM_JOB_NUM_NODES} $*"
