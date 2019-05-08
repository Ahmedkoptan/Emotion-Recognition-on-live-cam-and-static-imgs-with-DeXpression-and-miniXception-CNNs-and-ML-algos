#!/bin/bash

#SBATCH --nodes=8 			#number of compute nodes
#SBATCH -n 112 			#number of CPU cores to reserve on this compute node

#SBATCH --time=100:00:00
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=akobtan@asu.edu # send-to address

module purge    # Always purge modules to ensure a consistent environment

module load python/2.7.12-cidsegpu

export OMP_NUM_THREADS=112
export OMP_DYNAMIC=TRUE

python run_fer2013.py