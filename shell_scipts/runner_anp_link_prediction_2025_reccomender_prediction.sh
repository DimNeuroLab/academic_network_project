#!/bin/bash
# Get command line arguments

#learning_rate = float(sys.argv[1])
#infosphere_type = int(sys.argv[2])
#infosphere_parameters = sys.argv[3]
#only_new = sys.argv[4].lower() == 'true'
#edge_number = int(sys.argv[5])
#drop_percentage = float(sys.argv[6])
#sorted_flag = sys.argv[8].lower() == 'true'
#embedding_hindsight = sys.argv[9].lower() == 'true'
#GT_infosphere_type = int(sys.argv[10])
#GT_infosphere_parameters = sys.argv[11]

# infosphere embedding sorted
# GT infosphere type 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true false 2 10 
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true false 2 10
# GT infosphere type 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true false 3 5_2
# GT infosphere type 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true false 4 0
# GT infosphere type 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true false 5 0

# infosphere embedding random
# GT infosphere type 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false false 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false false 2 10
# GT infosphere type 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false false 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false false 3 5_2
# GT infosphere type 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false false 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false false 4 0
# GT infosphere type 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false false 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false false 5 0


# hindsight embedding random
# GT infosphere type 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false true 1 5
# GT infosphere type 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false true 2 10
# GT infosphere type 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false true 3 5_2
# GT infosphere type 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false true 4 0
# GT infosphere type 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 false true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 false true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 false true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 false true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 false true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 false true 5 0


# hindsight embedding sorted 
# GT infosphere type 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true true 1 5
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true true 1 5 
# GT infosphere type 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true true 2 10
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true true 2 10
# GT infosphere type 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true true 3 5_2
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true true 5 0
# GT infosphere type 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true true 4 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true true 4 0
# GT infosphere type 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 true true 5 0
CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 true true 5 0
