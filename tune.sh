#!/bin/bash
#
#CUDA_VISIBLE_DEVICES=0 python3 deploy_tune_main.py --oh_cols ccsr --n_iter 500 --models xgb --percase_cnt_vars ccsr cpt med123 --scaler none --gpu --n_jobs 50
#CUDA_VISIBLE_DEVICES=1 python3 deploy_tune_main.py --oh_cols cptgrp --n_iter 500 --models xgb --percase_cnt_vars ccsr cpt med123 --scaler none --gpu --n_jobs 50
#CUDA_VISIBLE_DEVICES=2 python3 deploy_tune_main.py --oh_cols pproc --n_iter 500 --models xgb --percase_cnt_vars ccsr cpt med123 --scaler none --gpu --n_jobs 20
#
#CUDA_VISIBLE_DEVICES=3 python3 deploy_tune_main.py --oh_cols cpt --n_iter 500 --models xgb --percase_cnt_vars ccsr cpt med123 --scaler none --gpu --n_jobs 10
#CUDA_VISIBLE_DEVICES=0 python3 deploy_tune_main.py --oh_cols ccsr cpt --n_iter 500 --models xgb --percase_cnt_vars ccsr cpt med123 --scaler none --gpu --n_jobs 10
#CUDA_VISIBLE_DEVICES=1 python3 deploy_tune_main.py --oh_cols ccsr pproc --n_iter 500 --models xgb --percase_cnt_vars ccsr cpt med123 --scaler none --gpu --n_jobs 10
#CUDA_VISIBLE_DEVICES=2 python3 deploy_tune_main.py --oh_cols ccsr cptgrp --n_iter 500 --models xgb --percase_cnt_vars ccsr cpt med123 --scaler none --gpu --n_jobs 15
#

iter=$1
model=$2
scaler=$3
prefix=$4

python3 deploy_tune_main.py --oh_cols ccsr --n_iter $iter --models $model --percase_cnt_vars ccsr cpt med123 --scaler $scaler --res_prefix prefix
python3 deploy_tune_main.py --oh_cols cptgrp --n_iter $iter --models $model --percase_cnt_vars ccsr cpt med123 --scaler $scaler --res_prefix prefix
python3 deploy_tune_main.py --oh_cols pproc --n_iter $iter --models $model --percase_cnt_vars ccsr cpt med123 --scaler $scaler --res_prefix prefix
python3 deploy_tune_main.py --oh_cols cpt --n_iter $iter --models $model --percase_cnt_vars ccsr cpt med123 --scaler $scaler --res_prefix prefix
python3 deploy_tune_main.py --oh_cols ccsr cptgrp --n_iter $iter --models $model --percase_cnt_vars ccsr cpt med123 --scaler $scaler --res_prefix prefix
python3 deploy_tune_main.py --oh_cols ccsr pproc --n_iter $iter --models $model --percase_cnt_vars ccsr cpt med123 --scaler $scaler --res_prefix prefix
python3 deploy_tune_main.py --oh_cols ccsr cpt --n_iter $iter --models $model --percase_cnt_vars ccsr cpt med123 --scaler $scaler --res_prefix prefix

