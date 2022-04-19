#!/bin/bash

#python3 deploy_tune_main.py --oh_cols ccsr --n_iter 100 --models rmf --percase_cnt_vars ccsr cpt --scaler none
#python3 deploy_tune_main.py --oh_cols ccsr cpt --n_iter 100 --models rmf --percase_cnt_vars ccsr cpt --scaler none
#python3 deploy_tune_main.py --oh_cols ccsr cptgrp --n_iter 100 --models rmf --percase_cnt_vars ccsr cpt --scaler none
#python3 deploy_tune_main.py --oh_cols ccsr pproc --n_iter 100 --models rmf --percase_cnt_vars ccsr cpt --scaler none


python3 deploy_tune_main.py --oh_cols ccsr --n_iter 5 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols ccsr cpt --n_iter 5 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols ccsr cptgrp --n_iter 5 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none
python3 deploy_tune_main.py --oh_cols ccsr pproc --n_iter 5 --models rmf --percase_cnt_vars ccsr cpt med123 --scaler none

