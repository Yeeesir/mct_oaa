#! /bin/bash
T=`date +%m%d%H%M`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
log_dir='test_animals_log'
if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi
python $ROOT/tools/test_baseline_animals.py --gpu 0 --transductive True --flip True  --n_shot 1 --n_train_class 15   |tee ./test_animals_log/test.animal.std.log.$T
