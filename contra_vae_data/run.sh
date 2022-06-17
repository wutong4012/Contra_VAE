#!/bin/sh

#SBATCH -J process_data # 作业名是test
#SBATCH -p batch # 提交到batch 分区
#SBATCH -N 1 # 申请使用一个节点
## SBATCH -w dgx039
#SBATCH --ntasks=1
## SBATCH --requeue # 重新提交
## SBATCH --qos=preemptive # 抢占

#SBATCH --cpus-per-task=128 # 申请CPU 核心
## SBATCH --gres=gpu:1 # 申请GPU 卡
#SBATCH -t 100-00:00:00 #
#SBATCH -o log.%x.job_%j
#SBATCH -e error

export CUDA_VISIBLE_DEVICES=6
SINGULARITY_PATH=/cognitive_comp/wutong/contra_vae/py21.11-py3_cont_vae.sif

singularity exec --nv -B /cognitive_comp:/cognitive_comp $SINGULARITY_PATH python /cognitive_comp/wutong/fs_datasets/contra_vae_wudao_180g/generate_load.py
