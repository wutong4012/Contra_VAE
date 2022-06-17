#!/bin/bash

SINGULARITY_PATH=/cognitive_comp/wutong/contra_vae/py21.11-py3_cont_vae.sif
srun -N 1 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=9 -e job_out/%x-%j.err -o job_out/%x-%j.log singularity exec --nv -B /cognitive_comp/wutong:/cognitive_comp/wutong $SINGULARITY_PATH python /cognitive_comp/wutong/contra_vae/main.py
