#! /bin/bash
T=`date +%m%d%H%M`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
log_dir='test_log'
if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi
python $ROOT/tools/test_baseline.py --gpu 1 --transductive True --flip True  --n_shot 1 --n_train_class 15   |tee ./test_log/train.log.$T