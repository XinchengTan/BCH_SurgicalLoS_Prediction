#!/bin/bash

python3 deploy_tune_main.py --oh_cols ccsr --n_iter 500 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols ccsr cpt --n_iter 500 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols ccsr cptgrp --n_iter 500 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols ccsr pproc --n_iter 500 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols pproc --n_iter 500 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols cpt --n_iter 500 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols cptgrp --n_iter 500 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none

