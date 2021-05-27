import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from utils.backbone.resnet12 import ResNet12
from utils.backbone.conv256 import ConvNet as ConvNet_256
from utils.backbone.conv128 import ConvNet as ConvNet_128
from utils.backbone.conv64 import ConvNet as ConvNet_64

class Runner(object):
    def __init__(self, nb_class_train, nb_class_test,  input_size, n_shot, n_query,
                 backbone='ResNet-12', flip=True, drop=True):

        self.nb_class_train = nb_class_train
        self.nb_class_test = nb_class_test
        self.input_size = input_size
        self.n_shot = n_shot
        self.n_query = n_query
        self.is_transductive = True
        self.flip = flip 
        self.drop = drop 

        # create model
        if backbone == 'ResNet-12':
            self.model = ResNet12(with_drop=drop)
        elif backbone == 'ConvNet-64':
            self.model = ConvNet_64(with_drop=drop)
        elif backbone == 'ConvNet-128':
            self.model = ConvNet_128(with_drop=drop)
        elif backbone == 'ConvNet-256':
            self.model = ConvNet_256(with_drop=drop)

        self.model.cuda()
        self.loss = nn.CrossEntropyLoss()

    def set_optimizer(self, learning_rate, weight_decay_rate):

        self.optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': weight_decay_rate}],
                                   lr = learning_rate, momentum=0.9, nesterov=True)

    def compute_accuracy(self, t_data, prob):
        t_est = torch.argmax(prob, dim=1)

        return (t_est == t_data)


    def make_protomap(self, support_set, nb_class):
        # 利用支持样本特征计算均值得到类别原型
        B, C, W, H = support_set.shape
        protomap = support_set.reshape(self.n_shot, nb_class, C, W, H)
        protomap = protomap.mean(dim=0)

        return protomap

    def make_input(self, images):
        # 将np形式的数据转换成torch
        images = np.stack(images)
        images = torch.Tensor(images).cuda()
        images = images.view(images.size(0), 84, 84, 3)
        images = images.permute(0, 3, 1, 2)
 
        return images

    def element_wise_scale(self, set):
        # 模型的backbone后面还有conv和全连接层
        x = self.model.conv1_ls(set)
        x = self.model.bn1_ls(x)
        x = self.model.relu(x)

        x = x.reshape(x.size(0), -1)
        x = self.model.fc1_ls(x)
        x = F.softplus(x)

        return x

    def add_query(self, support_set, query_set, prob, nb_class):
        # 在transductive设定下，将查询样本加到原型上，修正原型
        B, C, W, H = support_set.shape
        per_class = support_set.reshape(self.n_shot, nb_class, C, W, H)

        for i in range(nb_class):
            ith_prob = prob[:,i].reshape(prob.size(0), 1, 1, 1)
            ith_map = torch.cat((per_class[:,i], query_set*ith_prob), dim=0)
            ith_map = torch.sum(ith_map, dim=0, keepdim=True)/(ith_prob.sum()+self.n_shot)
            if i == 0: protomap = ith_map
            else: protomap = torch.cat((protomap, ith_map), dim=0)

        return protomap

    def norm_flatten(self, set):
        set = torch.flatten(set, start_dim=1)
        set = F.normalize(set, dim=1)

        return set

    def flip_key(self, images):
        # 得到翻转后的图像，作为简单的数据增强
        self.model.eval()
        with torch.no_grad():
            flipped_key = self.model(torch.flip(images, dims=[3]))
            return flipped_key

    def train_transduction(self, original_key, flipped_key, nb_class, iters=1):

        if not self.is_transductive: iters = 0
        nb_key = 2 if self.flip else 1
        prob_list = []
        # 此循环用于生成查询样本的confidence权重prob_list
        for iter in range(iters):#通常iters为1
            prob_sum = 0
            for i in range(nb_key): #transductive设定下nb_key为2
                if i != nb_key - 1: key_list = flipped_key
                else: key_list = original_key
                for idx, key in enumerate(key_list):
                    support_set = key[:nb_class * self.n_shot]# 支持样本为key中前nb_class * self.n_shot个
                    query_set = key[nb_class * self.n_shot:]
                    # Make Protomap
                    if iter == 0: protomap = self.make_protomap(support_set, nb_class) #首次生成原型
                    else: protomap = self.add_query(support_set, query_set, prob_list[iter-1], nb_class) #第二次将查询样本特征叠加到原型上
                    # Element-wise length scaling
                    if idx == 0:
                        s_q = self.element_wise_scale(query_set) #self.element_wise_scale()将backbone输出的特征输入后续的全连接层
                        s_p = self.element_wise_scale(protomap)
                    query_NF = self.norm_flatten(query_set) / s_q 
                    proto_NF = self.norm_flatten(protomap) / s_p
                    # Calculate distance
                    distance = query_NF.unsqueeze(1) - proto_NF 
                    distance = distance.pow(2).sum(dim=2)
                    prob = F.softmax(-distance, dim=1)
                    prob_sum += prob / (nb_key * len(key_list))
            prob_list.append(prob_sum)
        

        key = original_key[0]
        support_set = key[:nb_class * self.n_shot]
        query_set = key[nb_class * self.n_shot:]

        # 利用上面生成的每个查询样本预测的伪标签进行tranduction
        protomap = None
        if self.is_transductive:
            protomap = self.add_query(support_set, query_set, prob_list[-1], nb_class)
        elif not self.is_transductive:
            protomap = self.make_protomap(support_set, nb_class)

        s_p = self.element_wise_scale(protomap)
        scaled_proto = self.norm_flatten(protomap) / s_p
        #输出修正后的原型
        return scaled_proto

    def train(self, images, labels):
        # baseline的训练函数
        nb_class = self.nb_class_train
        images = self.make_input(images)
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()
        flipped_key = self.flip_key(images) if self.flip else None

        self.model.train()
        original_key = self.model(images)
        key = original_key[0]

        # pixel-wise classification
        key_DC = key[nb_class * self.n_shot:]
        key_DC = key_DC.reshape(key_DC.size(0), key_DC.size(1), -1)
        key_DC = key_DC.permute(0, 2, 1)
        #这个原型是dense原型，不是类别原型
        #利用一个可学习的向量model.weight.weight作为不同空间未知的原型
        #在训练中计算样本不同位置与这个向量的差并梯度回传更新这个向量的参数
        prototype = self.model.weight.weight
        
        #dense loss就是特征每个位置的二范数损失
        loss_dense = 0
        distance = key_DC.unsqueeze(2) - prototype
        distance = distance.pow(2).sum(dim=3)
        for i in range(distance.size(1)):
            loss_dense += self.loss(-distance[:,i], labels_DC[nb_class * self.n_shot:])/distance.size(1)

        #instance loss 是通过距离度量预测得到类别后计算的交叉熵loss
        # instance-wise classification
        labels_IC = tuple([i for i in range(nb_class)]) * (self.n_query)
        labels_IC = torch.tensor(labels_IC, dtype=torch.long).cuda()
        #make prototype
        scaled_proto = self.train_transduction(original_key, flipped_key, nb_class, iters=1)
        query_set = key[nb_class * self.n_shot:]
        s_q = self.element_wise_scale(query_set)
        scaled_query = self.norm_flatten(query_set) / s_q

        distance = scaled_query.unsqueeze(1) - scaled_proto
        distance = distance.pow(2).sum(dim=2)

        loss_instance = self.loss(-distance, labels_IC)

        loss = 0
        loss += 1 * loss_dense
        loss += 1/5 * loss_instance
        # 两种loss进行组合
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def train_without_dense(self, images, labels):
        # 相比于train，没有dense loss
        nb_class = self.nb_class_train
        images = self.make_input(images)
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()
        flipped_key = self.flip_key(images) if self.flip else None

        self.model.train()
        original_key = self.model(images)
        key = original_key[0]


        # instance-wise classification
        labels_IC = tuple([i for i in range(nb_class)]) * (self.n_query)
        labels_IC = torch.tensor(labels_IC, dtype=torch.long).cuda()
        #make prototype
        scaled_proto = self.train_transduction(original_key, flipped_key, nb_class, iters=1)
        query_set = key[nb_class * self.n_shot:]
        s_q = self.element_wise_scale(query_set)
        scaled_query = self.norm_flatten(query_set) / s_q

        distance = scaled_query.unsqueeze(1) - scaled_proto
        distance = distance.pow(2).sum(dim=2)

        loss_instance = self.loss(-distance, labels_IC)

        loss = 0
        loss += 1/5 * loss_instance

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def train_without_dense_merge(self, input1,input2, labels,first=True):
        # 相比于train，没有dense loss
        # 同时输入两个分支的特征input1,input2
        # 在训练阶段 支持样本特征为input1和input2的均值
        # first=True时，查询样本特征为input1；first=False时，查询样本特征为input2
        nb_class = self.nb_class_train
        input1 = self.make_input(input1)
        input2 = self.make_input(input2)
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()
        flipped_key1 = self.flip_key(input1) if self.flip else None
        flipped_key2 = self.flip_key(input2) if self.flip else None
        flipped_key = []
        assert len(flipped_key1)==1
        flipped_key.append(flipped_key1[0])
        flipped_key[0] = (flipped_key1[0]+flipped_key2[0])/2

        self.model.train()
        original_key1 = self.model(input1)
        original_key2 = self.model(input2)
        original_key = []
        original_key.append(original_key1[0])
        original_key[0] = (original_key1[0] + original_key2[0])/2
        key = original_key[0]


        # instance-wise classification
        labels_IC = tuple([i for i in range(nb_class)]) * (self.n_query)
        labels_IC = torch.tensor(labels_IC, dtype=torch.long).cuda()
        #make prototype
        scaled_proto = self.train_transduction(original_key, flipped_key, nb_class, iters=1)
        
        if first:
            query_set = original_key1[0][nb_class * self.n_shot:]
        else:
            query_set = original_key2[0][nb_class * self.n_shot:]

        s_q = self.element_wise_scale(query_set)
        scaled_query = self.norm_flatten(query_set) / s_q

        distance = scaled_query.unsqueeze(1) - scaled_proto
        distance = distance.pow(2).sum(dim=2)

        loss_instance = self.loss(-distance, labels_IC)

        loss = 0
        loss += 1/5 * loss_instance

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def train_merge(self, input1,input2, labels,first=True):
        # 有dense loss
        # 同时输入两个分支的特征input1,input2
        # 在训练阶段 支持样本特征为input1和input2的均值
        # first=True时，查询样本特征为input1；first=False时，查询样本特征为input2
        nb_class = self.nb_class_train
        input1 = self.make_input(input1)
        input2 = self.make_input(input2)

        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()

        flipped_key1 = self.flip_key(input1) if self.flip else None
        flipped_key2 = self.flip_key(input2) if self.flip else None
        flipped_key = []
        flipped_key.append(flipped_key1[0])
        flipped_key[0] = (flipped_key1[0]+flipped_key2[0])/2

        self.model.train()

        original_key1 = self.model(input1)
        original_key2 = self.model(input2)
        original_key = []
        original_key.append(original_key1[0])
        original_key[0] = (original_key1[0] + original_key2[0])/2

        key = original_key[0]

        # pixel-wise classification
        key_DC = key[nb_class * self.n_shot:]
        key_DC = key_DC.reshape(key_DC.size(0), key_DC.size(1), -1)
        key_DC = key_DC.permute(0, 2, 1)
        prototype = self.model.weight.weight

        loss_dense = 0
        distance = key_DC.unsqueeze(2) - prototype
        distance = distance.pow(2).sum(dim=3)
        for i in range(distance.size(1)):
            loss_dense += self.loss(-distance[:,i], labels_DC[nb_class * self.n_shot:])/distance.size(1)

        # instance-wise classification
        labels_IC = tuple([i for i in range(nb_class)]) * (self.n_query)
        labels_IC = torch.tensor(labels_IC, dtype=torch.long).cuda()
        #make prototype
        scaled_proto = self.train_transduction(original_key, flipped_key, nb_class, iters=1)
        
        if first:
            query_set = original_key1[0][nb_class * self.n_shot:]
        else:
            query_set = original_key2[0][nb_class * self.n_shot:]
        
        s_q = self.element_wise_scale(query_set)
        scaled_query = self.norm_flatten(query_set) / s_q

        distance = scaled_query.unsqueeze(1) - scaled_proto
        distance = distance.pow(2).sum(dim=2)

        loss_instance = self.loss(-distance, labels_IC)

        loss = 0
        loss += 1 * loss_dense
        loss += 1/5 * loss_instance

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def evaluate(self, images, labels):
        # baseline的评估函数
        nb_class = self.nb_class_test
        images = self.make_input(images)
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        self.model.eval()
        with torch.no_grad():
            flipped_key = self.model(torch.flip(images, dims=[3]))
            original_key = self.model(images)

            # iteration= 11说明有11轮的预测伪标签再修正原型的过程
            iteration= 11 if self.is_transductive else 1
            nb_key = 2 if self.flip else 1
            prob_list = []
            for iter in range(iteration):
                prob_sum = 0
                for i in range(nb_key):
                    if i != nb_key - 1:  key_list = flipped_key
                    else: key_list = original_key
                    for idx, key in enumerate(key_list):
                        support_set = key[:nb_class * self.n_shot]
                        query_set = key[nb_class * self.n_shot:]
                        # Make Protomap
                        if iter == 0: protomap = self.make_protomap(support_set, nb_class)
                        else: protomap = self.add_query(support_set, query_set, prob_list[iter-1], nb_class)
                        if idx == 0:
                            s_q = self.element_wise_scale(query_set)
                            s_p = self.element_wise_scale(protomap)
                        # Element-wise Scaling
                        query_NF = self.norm_flatten(query_set) / s_q
                        proto_NF = self.norm_flatten(protomap) / s_p
                        # Calculate Distance
                        distance = query_NF.unsqueeze(1) - proto_NF
                        distance = distance.pow(2).sum(dim=2)
                        prob = F.softmax(-distance, dim=1)
                        prob_sum += prob

                prob_list.append(prob_sum / (nb_key * len(key_list)))

            prob = prob_list[-1]
            acc = self.compute_accuracy(labels[nb_class * self.n_shot:], prob)
            prob = prob.data.cpu().numpy()

            return acc, prob, labels[nb_class*self.n_shot:]

    def evaluate_merge(self, input1,input2, labels,query_idx=1,spt_idx=2):
        # 将input1和input2特征融合，进行评估
        # spt_idx=1:spt为rgb || spt_idx=2:spt为(rgb+rgb_modify) || spt_idx=2:spt为rgb_modify
        # query_idx=1:qry为rgb || query_idx=2:qry为rgb_modify
        nb_class = self.nb_class_test
        input1 = self.make_input(input1)
        input2 = self.make_input(input2)
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        self.model.eval()
        with torch.no_grad():
            flipped_key1 = self.model(torch.flip(input1, dims=[3]))
            flipped_key2 = self.model(torch.flip(input2, dims=[3]))
            original_key1 = self.model(input1)
            original_key2 = self.model(input2)
            
            ###tmp pickle save
            # import pickle
            # path_pkl1 = '/home/shenyq/process/THUpaper/fsl_tsne/pkl_save/rgb5.pickle'
            # path_pkl2 = '/home/shenyq/process/THUpaper/fsl_tsne/pkl_save/rgb_mask5.pickle'
            # with open(path_pkl1,'wb') as fp:
            #     pickle.dump(original_key1[0],fp,protocol = pickle.HIGHEST_PROTOCOL)
            # with open(path_pkl2,'wb') as fp:
            #     pickle.dump(original_key2[0],fp,protocol = pickle.HIGHEST_PROTOCOL)
            ###
            

            iteration= 11 if self.is_transductive else 1
            nb_key = 2 if self.flip else 1
            prob_list = []
            for iter in range(iteration):
                prob_sum = 0
                for i in range(nb_key):
                    if i != nb_key - 1:  
                        key_list1 = flipped_key1
                        key_list2 = flipped_key2
                    else: 
                        key_list1 = original_key1
                        key_list2 = original_key2

                    for idx in range(len(key_list1)):
                        if spt_idx==2:
                            support_set = (key_list1[idx][:nb_class * self.n_shot]+key_list2[idx][:nb_class * self.n_shot])/2
                        elif spt_idx==1:
                            support_set = key_list1[idx][:nb_class * self.n_shot]
                        elif spt_idx==3:
                            support_set = key_list2[idx][:nb_class * self.n_shot]
                        else:
                            print('spt_idx error')
                            exit(1)
                        if query_idx==1:
                            query_set = key_list1[idx][nb_class * self.n_shot:]
                        elif query_idx==2:
                            query_set = key_list2[idx][nb_class * self.n_shot:]
                        else:
                            print('qry_idx error')
                            exit(1)
                        # Make Protomap
                        if iter == 0: protomap = self.make_protomap(support_set, nb_class)
                        else: protomap = self.add_query(support_set, query_set, prob_list[iter-1], nb_class)
                        if idx == 0:
                            s_q = self.element_wise_scale(query_set)
                            s_p = self.element_wise_scale(protomap)
                        # Element-wise Scaling
                        query_NF = self.norm_flatten(query_set) / s_q
                        proto_NF = self.norm_flatten(protomap) / s_p
                        # Calculate Distance
                        distance = query_NF.unsqueeze(1) - proto_NF
                        distance = distance.pow(2).sum(dim=2)
                        prob = F.softmax(-distance, dim=1)
                        prob_sum += prob

                prob_list.append(prob_sum / (nb_key * len(key_list1)))

            prob = prob_list[-1]
            acc = self.compute_accuracy(labels[nb_class * self.n_shot:], prob)
            prob = prob.data.cpu().numpy()

            return acc, prob, labels[nb_class*self.n_shot:]

    