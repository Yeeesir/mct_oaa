# MCT
baseline from git@github.com:seongmin-kye/MCT.git

Train:
cd mini/exp_mini/exp_base_trans
sh train_trans.sh


Files need to read in detail:

	MCT/mini/utils/generator:
		generators_train_mask.py  --add mask when load image;
		generators_test_mask.py --add mask when load image;
		generators_train_mask_bg.py --add mask and generated background when load image;
		generators_test_mask_bg.py --add mask and generated background when load image;
		Note: 
		mask--fg is [255,255,255],bg is [0,0,0]
		generated background--fg is filled with bg, which is used to generated new samples by add fg directly;
		
	/MCT/mini/utils:
		model_branch_attenion.py --mask*image on 2 branch(instance loss branch and dense loss branch)
		model_ba_denseins.py --add some args to set weight of each branch and choose whether to use mask on each branch;
		Note:
		These args keep default states(False), because multipying mask with image didn't get good performance.
	
	/MCT/mini/tools:
		train_branch_attention.py --add mask*image and some new args
		eval_ba_mini_bg.py --eval model with mini
		eval_ba_coco_bg.py --eval model with coco
		Note:
		1. _bg means generate new samples with bgs when evaling;
		2. _att means multiply mask with images during train stage;
		3. other files are familiar with these except that they may not have masks;
	
	MCT/mini/exp_mini
		Different settings of experiments can be managed here, for example:
		train_trans.sh 
		--transductive True --flip True --drop True --n_shot 1 
		args are set in sh file.
		

