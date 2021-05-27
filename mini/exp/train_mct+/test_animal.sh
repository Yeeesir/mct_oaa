#! /bin/bash
T=`date +%m%d%H%M`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
log_dir='eval_animal_log'
if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi
python $ROOT/tools/test_task1_animals5.py --gpu 1 --transductive True --flip True  --n_shot 1 --n_train_class 15   |tee ./eval_animal_log/eval.animal.log.$T
