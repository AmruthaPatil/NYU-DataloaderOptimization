#!/bin/bash
#SBATCH --job-name=l2_c2
#SBATCH --output=%x.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu

module purge

singularity exec --nv \
            --overlay /scratch/username/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; cd /scratch/username/lab2; python c1.py;"