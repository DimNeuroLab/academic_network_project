#!/bin/bash
# hindsight embedding sorted 
python3 coauthor_prediction/create_GT/create_gt_opt_rnu_r.py 1 5 false true true
python3 coauthor_prediction/create_GT/create_gt_opt_rnu_r.py 4 0 false true true
python3 coauthor_prediction/create_GT/create_gt_opt_rnu_r.py 5 0 false true true

# GT infosphere type 1 5
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 0 5 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 1 5 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 2 10 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 3 [5,2] false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 4 0 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 5 0 false 50 0 0 true true 1 5 
# GT infosphere type 4 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 0 5 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 1 5 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 2 10 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 3 [5,2] false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 4 0 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 5 0 false 50 0 0 true true 4 0
# GT infosphere type 5 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 0 5 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 1 5 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 2 10 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 3 [5,2] false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 4 0 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=3 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_rnu_r.py 0.00001 5 0 false 50 0 0 true true 5 0
