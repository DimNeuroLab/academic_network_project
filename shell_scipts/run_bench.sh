#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_counter_rmse.py 0.001 0  0 false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_counter_poisson.py 0.001 0  0 false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_counter_negbinom.py 0.001 0  0 false -1 0 0

python3 coauthor_prediction/anp_link_prediction_co_author_hgt_counter_rmse.py 0.001 0  0 true -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_counter_poisson.py 0.001 0  0 true -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_counter_negbinom.py 0.001 0  0 true -1 0 0
















