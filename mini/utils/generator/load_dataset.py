import pickle as pkl
import numpy as np
import os
import cv2
def load_pickle(data_file):
    try:
        with open(data_file, 'rb') as fo:
            data = pkl.load(fo)
            print(1)
        return data
    except:
        with open(data_file, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data
def buildLabelIndex( labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

def load_data(data_file):
    dataset = load_pickle(data_file)
    data = dataset['data']
    labels = dataset['labels']
    label2ind = buildLabelIndex(labels)
    return {key: np.array(data[val]) for (key, val) in label2ind.items()}

def load_mask(mask_file):
    dataset = load_pickle(mask_file)
    mask = dataset['mask']
    labels = dataset['labels']
    label2ind = buildLabelIndex(labels)
    return {key: np.array(mask[val]) for (key, val) in label2ind.items()}

if __name__ =='__main__':
    data_path = '/home/shenyq/zsl/MCT_DFMN/mini_ImageNet/data/miniImageNet'
    train_path = data_path + '/miniImageNet_category_split_test.pickle'
    # data_path = '/home/shenyq/zsl/MCT_DFMN/mini_ImageNet/data/miniImageNetMask/'
    # train_path = data_path + '/miniImageNet_category_split_train_phase_train.pickle'
    # train_path = data_path + 'mini_test_maskv2.pickle'
    data = load_data(train_path)
    # mask = load_mask(train_path)
    img_save_path_root = '/home/shenyq/zsl/MCT_DFMN/mini_ImageNet/data/miniImageNetMask/test/'
    
    for i in range(len(data)):
        img_save_dir = img_save_path_root +str(i)+'/'
        if  not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        for ii in range(5):
            img_save_path = img_save_dir + str(ii)+'.jpg'
            img = data[80+i][ii,:,:,:]
            #img_resize = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_save_path,img)
            print("%d , %d"%(i,ii))
            
                

    print(1)