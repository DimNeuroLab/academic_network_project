#!/bin/bash

# Usage:
# ./run.sh <macroblock> <cuda_device>
# Example:
# ./run.sh sorted 0

MACROBLOCK=$1
CUDA_DEVICE=$2

if [[ -z "$MACROBLOCK" || -z "$CUDA_DEVICE" ]]; then
  echo "Usage: $0 <macroblock: sorted|random|hindsight_sorted|hindsight_random> <cuda_device: 0|1>"
  exit 1
fi

run_block() {
  local sorted_flag=$1
  local hindsight_flag=$2
  
  if [ "$hindsight_flag" = true ]; then
    CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 $sorted_flag $hindsight_flag 1 5
    CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 $sorted_flag $hindsight_flag 1 5
    CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 $sorted_flag $hindsight_flag 1 5
    CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 $sorted_flag $hindsight_flag 1 5
    CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 $sorted_flag $hindsight_flag 1 5
    CUDA_VISIBLE_DEVICES=1 python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 $sorted_flag $hindsight_flag 1 5
  fi

  # GT infosphere type 2 10
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 $sorted_flag $hindsight_flag 2 10
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 $sorted_flag $hindsight_flag 2 10
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 $sorted_flag $hindsight_flag 2 10
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 $sorted_flag $hindsight_flag 2 10
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 $sorted_flag $hindsight_flag 2 10
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 $sorted_flag $hindsight_flag 2 10
  
  # GT infosphere type 3 5_2
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 $sorted_flag $hindsight_flag 3 5_2
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 $sorted_flag $hindsight_flag 3 5_2
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 $sorted_flag $hindsight_flag 3 5_2
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 $sorted_flag $hindsight_flag 3 5_2
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 $sorted_flag $hindsight_flag 3 5_2
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 $sorted_flag $hindsight_flag 3 5_2
  
  # GT infosphere type 4 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 $sorted_flag $hindsight_flag 4 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 $sorted_flag $hindsight_flag 4 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 $sorted_flag $hindsight_flag 4 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 $sorted_flag $hindsight_flag 4 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 $sorted_flag $hindsight_flag 4 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 $sorted_flag $hindsight_flag 4 0
  
  # GT infosphere type 5 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 $sorted_flag $hindsight_flag 5 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 $sorted_flag $hindsight_flag 5 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 $sorted_flag $hindsight_flag 5 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 $sorted_flag $hindsight_flag 5 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 $sorted_flag $hindsight_flag 5 0
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 5 0 false 50 0 0 $sorted_flag $hindsight_flag 5 0
}

case $MACROBLOCK in
  sorted)
    run_block true false
    ;;
  random)
    run_block false false
    ;;
  hindsight_sorted)
    run_block true true
    ;;
  hindsight_random)
    run_block false true
    ;;
  *)
    echo "Unknown macroblock: $MACROBLOCK"
    echo "Valid options: sorted | random | hindsight_sorted | hindsight_random"
    exit 1
    ;;
esac
