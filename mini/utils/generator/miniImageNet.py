"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot \
                              and https://github.com/kjunelee/MetaOptNet
"""
import numpy as np
import random
import pickle as pkl
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import PIL.Image as Image

class miniImageNetGenerator(Dataset):

    def __init__(self, data_file,mode='train'):
        super(miniImageNetGenerator, self).__init__()
        self.data = self._load_data(data_file)
        self.mode = mode
        self.imgs_list,self.labels_list = self.make_dataset()
        print(1)
        #self.mask = self._load_mask(self.data_file)
        if mode=='train':
            self.transform = transforms.Compose([lambda x: x,
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                    # transforms.RandomHorizontalFlip(),
                                                    # transforms.RandomCrop(32, 4),
                                                    ])
        else:
            self.transform = transforms.Compose([lambda x: x,
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                    ])


    def make_dataset(self):
        imgs_list =[]
        labels_list = []
        for cat in self.data:
            imgs = self.data[cat][0]
            for idx in range(imgs.shape[0]):
                if self.mode=='train':
                    if idx<500:
                        imgs_list.append(imgs[idx,...])
                        labels_list.append(cat)
                else:
                    if idx>=500:
                        imgs_list.append(imgs[idx,...])
                        labels_list.append(cat)
        return imgs_list,labels_list

    def _load_data(self, data_file):
        dataset = self.load_data(data_file)
        data = dataset['data']
        mask = dataset['mask']
        labels = dataset['labels']
        label2ind = self.buildLabelIndex(labels)
        #return {key: np.array([data[val],mask[val]]) for (key, val) in label2ind.items()}

        return {key: [np.array(data[val]),np.array(mask[val])] for (key, val) in label2ind.items()}


    def load_data(self, data_file):
        try:
            with open(data_file, 'rb') as fo:
                data = pkl.load(fo)
            return data
        except:
            with open(data_file, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data

    def buildLabelIndex(self, labels):
        label2inds = {}
        for idx, label in enumerate(labels):
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

        return label2inds


    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self,idx):

        if self.mode=='train':
            img = Image.fromarray(self.imgs_list[idx])
            return self.transform(img),torch.tensor([self.labels_list[idx]])

        return self.transform(self.imgs_list[idx]),torch.tensor([self.labels_list[idx]])

if __name__ == '__main__':
    data_path = '/home/shenyq/zsl/MCT_DFMN/mini_ImageNet/data/miniImageNetMask'
    train_path = data_path + '/mini_train_maskv2.pickle'
    data = miniImageNetGenerator(train_path,mode='train')
    train_loader = torch.utils.data.DataLoader(data,batch_size=4, shuffle=True,\
        num_workers=0, pin_memory=True)
    for i,(img,label) in enumerate(train_loader):
        print(img.size())
    print(0)
