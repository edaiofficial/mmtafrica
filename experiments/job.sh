#!/bin/bash
#SBATCH --job-name=mmt
#SBATCH --gres=gpu:48gb:2
#SBATCH --cpus-per-gpu=18
#SBATCH --mem=180G          
#SBATCH --time=168:00:00         
#SBATCH --partition=long
#SBATCH --error=YOUR_HOME_PATH/slurmerror.txt
#SBATCH --output=YOUR_HOME_PATH/slurmoutput.txt


###########cluster information above this line
module load python/3.7
cd YOUR_HOME_PATH
source env/bin/activate 
#Comment below if you have already installed the packages
pip install -r requirements.txt
python mmt_bt.py