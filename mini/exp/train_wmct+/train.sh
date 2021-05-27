#! /bin/bash
T=`date +%m%d%H%M`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
log_dir='log'
if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi
python $ROOT/tools/train_wmct+.py --gpu 2 --transductive True --flip True  --n_shot 1 --n_train_class 15   |tee ./log/train.log.$T
