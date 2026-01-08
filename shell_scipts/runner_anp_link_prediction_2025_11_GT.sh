#!/bin/bash
# hindsight embedding sorted 
python3 coauthor_prediction/create_GT/create_gt_opt.py 1 5 false true true
#python3 coauthor_prediction/create_GT/create_gt_opt.py 2 10 false true true
#python3 coauthor_prediction/create_GT/create_gt_opt.py 3 5_2 false true true
python3 coauthor_prediction/create_GT/create_gt_opt.py 4 0 false true true
python3 coauthor_prediction/create_GT/create_gt_opt.py 5 0 false true true