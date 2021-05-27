#! /bin/bash
T=`date +%m%d%H%M`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
log_dir='log'
if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi
python $ROOT/tools/train_mct++.py --gpu 1 --transductive True --flip True  --n_shot 1 --n_train_class 15   |tee ./log/train.log.$T
