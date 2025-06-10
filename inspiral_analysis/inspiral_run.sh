#!/bin/bash
#SBATCH --time=13-1:00:00
#SBATCH --qos=long
#SBATCH --partition cpu,uri-cpu
#SBATCH --mem-per-cpu=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 
#SBATCH --output=gw150914_test_%j.log
#SBATCH --error=gw150914_test_%j.err
#SBATCH --mail-user=ashwin.girish@uri.edu
#SBATCH --mail-type=ALL

#
module load conda/latest
conda activate igwn-py310
#
date
#
python inspiral.py
#
date