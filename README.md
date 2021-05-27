# MCT
baseline from git@github.com:seongmin-kye/MCT.git

Train:
cd mini/exp_mini/exp_base_trans
sh train_trans.sh

Introduction to documents
	mct_oaa/mini/utils/generator:
		generators_test_mul5.py 测试任务生成
		generators_train_mul5.py 训练任务生成
		* baseline文件:
			generators_test.py 
			generators_test.py 
		* 无关紧要测试文件:
			load_dataset.py
			miniImageNet_full.py
			miniImageNet.py
		
	mct_oaa/mini/utils:
		model_mul.py 模型的选择，不同方法训练和测试的实现细节
			train() baseline的训练
			train_without_dense() wmct的训练
			train_without_dense_merge() 双分支网络的训练wmct+和wmct++都采用这个函数进行训练
			train_merge() mct+和mct++都采用这个函数进行训练
		model.py baseline使用的model文件
		
	mct_oaa/mini/tools:
		train_baseline.py mct的baseline训练
		train_baseline_nodense.py wmct的baseline训练
		train_mct+.py
		train_mct++.py
		train_wmct+.py
		train_wmct++.py
		test_baseline.py mct的baseline测试
		test_baseline_animals.py mct的baseline在animals5上测试
		test_task1.py ()+测试
		test_task1_animals5.py ()+测试
		test_task2.py ()++测试
		test_task2_animals5.py ()++测试
	
	mct_oaa/mini/exp
		baseline_dense/
		baseline_nodense/
		train_mct+/
		train_mct++/
		train_wmct+/
		train_wmct++/
			log/ 训练日志
			save/ 保存模型
			test_log/ 测试日志
			test_animals/ 在animals5上的测试日志

		
		

