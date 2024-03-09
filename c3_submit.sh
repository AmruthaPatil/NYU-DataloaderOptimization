#!/bin/bash
#SBATCH --job-name=l2_c3
#SBATCH --output=%x.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=ap7982@nyu.edu

module purge

singularity exec --nv \
            --overlay /scratch/ap7982/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; cd /scratch/ap7982/lab2; python c1.py --num-workers 0;\
                python c1.py --num-workers 4; python c1.py --num-workers 8; python c1.py --num-workers 12;\
                python c1.py --num-workers 16;"