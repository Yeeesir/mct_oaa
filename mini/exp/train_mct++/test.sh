#! /bin/bash
T=`date +%m%d%H%M`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
log_dir='eval_log'
if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi
python $ROOT/tools/test_task2.py --gpu 0 --transductive True --flip True  --n_shot 1 --n_train_class 15   |tee ./eval_log/eval.log.$T
