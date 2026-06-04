#!/bin/bash
# Get coauthor list
# CUDA_VISIBLE_DEVICES=2 python3 /data/sabrina/academic_network_project/anp_utils/get_author_from_recbole.py 10_LightGCN 5 0
# CUDA_VISIBLE_DEVICES=2 python3 /data/sabrina/academic_network_project/anp_utils/get_author_from_recbole.py 10_NAIS 4 0

# Create embeddings
# CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_embedding_generation_2026.py 0 0
# CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_embedding_generation_2026.py 1 5
# CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_embedding_generation_2026.py 2 10
# CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_embedding_generation_2026.py 3 [5,2]
# CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_embedding_generation_2026.py 4 0
# CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_embedding_generation_2026.py 5 0

# Create GTS
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/create_GT/create_gt_opt_rnu_history.py 1 5 false true false
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/create_GT/create_gt_opt_rnu_history.py 2 10 false true false
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/create_GT/create_gt_opt_rnu_history.py 3 [5,2] false true false
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/create_GT/create_gt_opt_rnu_history.py 4 0 false true false
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/create_GT/create_gt_opt_rnu_history.py 5 0 false true false

# infosphere embedding sorted
# GT infosphere type 2 10
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 0 5 false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 1 5 false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 2 10 false 50 0 0 true false 2 10 
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 3 [5,2] false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 4 0 false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 5 0 false 50 0 0 true false 2 10
# GT infosphere type 3 5_2
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 0 5 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 1 5 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 2 10 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 3 [5,2] false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 4 0 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 5 0 false 50 0 0 true false 3 5_2
# GT infosphere type 4 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 0 5 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 1 5 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 2 10 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 3 [5,2] false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 4 0 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 5 0 false 50 0 0 true false 4 0
# GT infosphere type 5 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 0 5 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 1 5 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 2 10 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 3 [5,2] false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 4 0 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=2 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model_cv_rnu_history.py 0.00001 5 0 false 50 0 0 true false 5 0

