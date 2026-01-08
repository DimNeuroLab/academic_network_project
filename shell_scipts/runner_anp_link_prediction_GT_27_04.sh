#!/bin/bash

# Check if the user provided an argument
if [ $# -eq 0 ]; then
  echo "Please provide a command argument."
  echo "Usage: $0 [GT_4_0 | GT_3_5_2 | GT_2_10]"
  exit 1
fi

COMMAND=$1

case $COMMAND in
  GT_4_0)
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 10 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,2] false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 4 0 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,10] false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 3 [5,10] false -1 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 50 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 2 50 false -1 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 1 5 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_improved_model.py 0.00001 0 5 false 50 0 0 4 1
    
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 10 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,2] false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 4 0 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,10] false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,10] false -1 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 50 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 50 false -1 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 1 5 false 50 0 0 4 1
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 0 5 false 50 0 0 4 1
    ;;

  GT_NEW)
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 10 false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,2] false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 4 0 false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 5 0 false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 50 false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 50 false -1 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,10] false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,10] false -1 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 1 5 false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 0 5 false 50 0 0 3 5_2
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 10 false 50 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,2] false 50 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 4 0 false 50 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 5 0 false 50 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 50 false 50 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 50 false -1 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,10] false 50 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,10] false -1 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 1 5 false 50 0 0 2 10
    python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 0 5 false 50 0 0 2 10
    ;;

  *)
    echo "Unknown command: $COMMAND"
    echo "Usage: $0 [GT_4_0 | GT_NEW]"
    exit 1
    ;;
esac
