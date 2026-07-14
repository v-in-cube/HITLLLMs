#!/bin/bash
#SBATCH --job-name=gpt55
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=varvara.voinarovska@astrazeneca.com
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-15:00:00
#SBATCH --output=gpt55-%j.log
#SBATCH --error=gpt55-%j.log

source ~/conda_init.sh
conda activate stats_hitl_llms
cd /projects/mai/se_mai/users/kjkh840_varvara/HITL_vs_LLM/HITLLLMs_new_llms/llms_querying/
python3 gpt55.py
