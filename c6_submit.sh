#!/bin/bash
#SBATCH --job-name=l2_c6
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
            /bin/bash -c "source /ext3/env.sh; cd /scratch/ap7982/lab2;\
                python c1.py --num-workers 4 --optimizer "sgd"; \
                python c1.py --num-workers 4 --optimizer "nesterov"; \
                python c1.py --num-workers 4 --optimizer "adagrad";\
                python c1.py --num-workers 4 --optimizer "adadelta";\
                python c1.py --num-workers 4 --optimizer "adam";"