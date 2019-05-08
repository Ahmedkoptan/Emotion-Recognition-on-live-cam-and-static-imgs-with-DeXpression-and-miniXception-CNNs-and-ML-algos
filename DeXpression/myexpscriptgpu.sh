#!/bin/bash

#SBATCH --nodes=8 			#number of compute nodes
#SBATCH -n 112 			#number of CPU cores to reserve on this compute node

#SBATCH -p physicsgpu1		#Use cidsegpu1 partition
#SBATCH -q wildfire		#Run job under wildfire QOS queue

#SBATCH --gres=gpu:4		#cidsegpu1 : 4, physi

#SBATCH --time=100:00:00
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=akobtan@asu.edu # send-to address

module purge    # Always purge modules to ensure a consistent environment

module load python/2.7.12-cidsegpu
##module load tensorflow/1.12-py3
module load cuda/9.0.176
module load cudnn/7.0
apt-get install --user cuda-cublas-[9.0.176]
pip install --user keras
pip3 install --user tensorflow
pip install --user TensorFlow
pip install --user tflearn




export OMP_NUM_THREADS=112
export OMP_DYNAMIC=TRUE


python DeXpression.py