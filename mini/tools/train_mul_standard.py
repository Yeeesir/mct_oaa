import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import cv2

import torch
from utils.generator.generators_train_mul5 import miniImageNetGenerator as train_loader
from utils.generator.generators_test_mul5 import miniImageNetGenerator as test_loader

from utils.model_mul import Runner

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Choice train or test.')
    parser.add_argument('--n_folder', type=int, default=0,
                        help='Number of folder.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number.')
    parser.add_argument('--backbone', type=str, default='ResNet-12',
                        help='Choice backbone such as ConvNet-64, ConvNet-128, ConvNet-256 and ResNet-12.')
    parser.add_argument('--initial_lr', type=float, default=1e-1,
                        help='Initial learning rate.')
    parser.add_argument('--first_decay', type=int, default=25000,
                        help='First decay step.')
    parser.add_argument('--second_decay', type=int, default=35000,
                        help='Second decay step.')
    parser.add_argument('--flip', type=str2bool, default=True,
                        help='Whether to inject data uncertainty.')
    parser.add_argument('--drop', type=str2bool, default=False,
                        help='Whether to inject model uncertainty.')
    parser.add_argument('--n_shot', type=int, default=1,
                        help='Number of support set per class in train.')
    parser.add_argument('--n_query', type=int, default=8,
                        help='Number of queries per class in train.')
    parser.add_argument('--n_test_query', type=int, default=15,
                        help='Number of queries per class in test.')
    parser.add_argument('--n_train_class', type=int, default=15,
                        help='Number of way for training episode.')
    parser.add_argument('--n_test_class', type=int, default=5,
                        help='Number of way for test episode.')
    parser.add_argument('--checkpoints', type=str, default='checkpoint',
                        help='checkpoints save path ')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #######################
    folder_num = args.n_folder

    # optimizer setting
    
    lrstep2 = args.second_decay
    lrstep1 = args.first_decay
    initial_lr = args.initial_lr

    # train episode setting
    n_shot=args.n_shot
    n_query=args.n_query
    nb_class_train = args.n_train_class

    # test episode setting
    n_query_test = args.n_test_query
    nb_class_test=args.n_test_class

    
    #data path 
    data_path = '/home/shenyq/data/mini/pickle/multi'
    train_path = data_path + '/mini_train.pickle'
    val_path = data_path + '/mini_val.pickle'
    test_path = data_path + '/mini_test.pickle'

    #save_path 保存训练好的模型
    save_path = 'save/checkpoints_' +  args.checkpoints
    if  not os.path.exists(save_path):
        os.makedirs(save_path)

    model_best= save_path + '/model_best.pth'
    
    # set up model
    # 通过模型设置训练细节  
    # 设置当前的任务为nb_class_train-way-n_shot任务
    # 使用args.backbone指定的backbone
    model = Runner(nb_class_train=nb_class_train, nb_class_test=nb_class_test, input_size=3*84*84,
                   n_shot=n_shot, n_query=n_query, backbone=args.backbone,
                   flip=args.flip, drop=args.drop)
    print('-----------------model building----------------')
    print(model)
    # 设置优化器
    model.set_optimizer(learning_rate=initial_lr, weight_decay_rate=5e-4)

    loss_h=[]
    accuracy_h_val=[]
    accuracy_h_test=[]

    acc_best=0
    epoch_best=0 #保存最后结果的epoch数

    max_iter = 50000
    val_iter = 300
    test_iter =100
    # start training
    # ----------------
    if args.is_train:
        # 生成nb_classes-way-n_shot-shot-n_query-query的训练任务，共max_iter个
        train_generator = train_loader(data_file=train_path, nb_classes=nb_class_train,
                                       nb_samples_per_class=n_shot + n_query, max_iter=max_iter)
        #images：原始图像
        #masks：前景分割结果
        #rgb_masks：前景物体
        #rgb_mask_mos：按照bbox切割出来的前景物体（不含背景）
        #rgb_mos：按照bbox切割出来的前景物体（包含背景） 
        #labels：类别标签
        for t, (images, masks,rgb_masks, rgb_mask_mos, rgb_mos, labels) in train_generator:
            # train
            # train_without_dense_merge(self, input1,input2, labels,first=True)
            # 此函数是没有transductive的
            # 训练阶段的支持样本融合了input1和input2的特征
            # 查询样本为input1或input2：当first=True，查询样本为input1；当first=False，查询样本为input2
            loss = model.train_without_dense_merge(images,rgb_mos, labels,first=False)
            loss_h.extend([loss.tolist()])
            print("Episode: %d / %d, Train Loss: %f "%(t,max_iter, loss))
            
            # 每2000个task后测试一下当前模型的性能
            if (t != 0) and (t % 2000 == 0):
                #生成测试任务，共val_iter个
                test_generator = test_loader(data_file=val_path, nb_classes=nb_class_test,
                                             nb_samples_per_class=n_shot+n_query_test, max_iter=val_iter)
                print('Evaluation in Validation data')
                scores = []
                for i, (images, masks,rgb_masks, rgb_mask_mos, rgb_mos, labels) in test_generator:
                    # evaluate_merge(self, input1,input2, labels,query_idx=1,spt_idx=2)
                    # 将input1和input2特征融合，进行评估
                    # spt_idx=1:spt为rgb || spt_idx=2:spt为(rgb+rgb_modify) || spt_idx=2:spt为rgb_modify
                    # query_idx=1:qry为rgb || query_idx=2:qry为rgb_modify
                    acc, prob, label = model.evaluate_merge(images,rgb_mos, labels,query_idx=2,spt_idx=3)
                    score = acc.data.cpu().numpy()
                    scores.append(score)

                print(('Accuracy {}-shot ={:.2f}%').format(n_shot, 100*np.mean(np.array(scores))))
                accuracy_t=100*np.mean(np.array(scores))
                #
                if acc_best < accuracy_t:
                    acc_best = accuracy_t
                    epoch_best=t
                    #保存训练阶段评估结果最好的模型
                    torch.save(model.model.state_dict(),model_best)
                accuracy_h_val.extend([accuracy_t.tolist()])
                
                del(acc)
                del(accuracy_t)

                if len(accuracy_h_val) >5:
                    print('***Average accuracy on past 10 test acc***')
                    print('Best epoch =',epoch_best,'Best {}-shot acc='.format(n_shot), acc_best)

            # 参考baseline的方法进行学习率衰减
            if (t != 0) & (t % lrstep1 == 0):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= 0.06
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')
            if (t != 0) & (t % lrstep2 == 0):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= 0.2
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')


    # 测试模型性能
    accuracy_h5=[]
    total_acc = []
    # 加载训练好的模型
    model.model.load_state_dict(torch.load(model_best))
    print('Evaluating the best {}-shot model...'.format(n_shot))
    # 评估10轮
    for i in range(10):
        # 生成test_iter个测试任务
        test_generator = test_loader(data_file=test_path, nb_classes=nb_class_test,
                                     nb_samples_per_class=n_shot+n_query_test, max_iter=test_iter)
        scores=[]
        for j, (images, masks,rgb_masks, rgb_mask_mos, rgb_mos, labels) in test_generator:
            #按照设定将双分支特征融合后进行性能评估
            acc, prob, label = model.evaluate_merge(images,rgb_mos, labels,query_idx=2,spt_idx=3)
            score = acc.data.cpu().numpy()
            scores.append(score)
            total_acc.append(np.mean(score) * 100)

        accuracy_t=100*np.mean(np.array(scores))
        accuracy_h5.extend([accuracy_t.tolist()])
        print(('100 episodes with 15-query accuracy: {}-shot ={:.2f}%').format(n_shot, accuracy_t))
        del(test_generator)
        del(acc)
        del(accuracy_t)

    stds = np.std(total_acc, axis=0)
    ci95 = 1.96 * stds / np.sqrt(len(total_acc))

    print(('Accuracy_test {}-shot ={:.2f}({:.2f})').format(n_shot, np.mean(accuracy_h5), ci95))