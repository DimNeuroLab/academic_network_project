#!/bin/bash
# python3 coauthor_prediction/anp_embedding_generation_2025_improved_model.py 0.00001 1 5 true -1 0 0
# python3 coauthor_prediction/anp_embedding_generation_2025_improved_model.py 0.00001 1 5 false -1 0 0

# python3 coauthor_prediction/anp_embedding_generation_2025_improved_model.py 0.00001 0 0 true -1 0 0
CUDA_VISIBLE_DEVICES=0 python3 coauthor_prediction/anp_embedding_generation_2025_improved_model.py 0.0001 0 0 false -1 0 0

# python3 coauthor_prediction/anp_embedding_generation_2025_improved_model.py 0.00001 0 0 true -1 0 0
CUDA_VISIBLE_DEVICES=0 python3 coauthor_prediction/anp_embedding_generation_2025_improved_model.py 0.0001 4 1 false -1 0 0