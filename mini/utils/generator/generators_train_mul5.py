
import numpy as np
import random
import pickle as pkl

class miniImageNetGenerator(object):

    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=10,
                  max_iter=None, xp=np):
        super(miniImageNetGenerator, self).__init__()
        self.data_file = data_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(self.data_file)
        #self.mask = self._load_mask(self.data_file)

    def _load_data(self, data_file):
        dataset = self.load_data(data_file)
        data = dataset['data']
        mask = dataset['mask']
        rgb_mask = dataset['rgb_mask']
        rgb_mask_mo = dataset['rgb_mask_mo']
        rgb_mo = dataset['rgb_mo']
        labels = dataset['labels']
        label2ind = self.buildLabelIndex(labels)
        #return {key: np.array([data[val],mask[val]]) for (key, val) in label2ind.items()}

        return {key: [np.array(data[val]), np.array(mask[val]), np.array(rgb_mask[val]),
                np.array(rgb_mask_mo[val]), np.array(rgb_mo[val])] for (key, val) in label2ind.items()}



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


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, masks, rgb_masks,rgb_mask_mos,rgb_mos, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, masks,rgb_masks,rgb_mask_mos,rgb_mos , labels)
        else:
            raise StopIteration()


    def augment(self, img):

        # random cropping
        npad = ((8, 8), (8, 8), (0, 0))
        img = np.pad(img, npad, 'constant', constant_values=(0.0))
        x = random.randint(0, 16)
        y = random.randint(0, 16)
        img = img[y:y + 84, x:x + 84]

        # random flipping
        flip_sign = random.randint(1,2)
        if flip_sign == 1:
            img = np.flip(img, 1)

        return img

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.data.keys()))
        pic_to_idx = {pic: i for i, pic in enumerate(picture_list)}
        sampled_characters = random.sample(self.data.keys(), nb_classes)

        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            label = pic_to_idx[char]
            _imgs = self.data[char][0]
            _mask = self.data[char][1]
            _rgb_mask = self.data[char][2]
            _rgb_mask_mo = self.data[char][3]
            _rgb_mo = self.data[char][4]

            save = _mask.copy()
            # cv2.imwrite('mask1.jpg',save)
            save[save<=10]=0
            save[save>10]=1
           
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            # labels_and_images.extend([(label, self.xp.array(self.augment(_imgs[i]/np.float32(255))),\
            #     self.xp.array(_mask[i]/np.float32(255))) for i in _ind])
            labels_and_images.extend([(label, self.xp.array(self.augment(_imgs[i]/np.float32(255))),
                self.xp.array(save[i]),self.xp.array(self.augment(_rgb_mask[i]/np.float32(255))),
                self.xp.array(self.augment(_rgb_mask_mo[i]/np.float32(255))),
                self.xp.array(self.augment(_rgb_mo[i]/np.float32(255)))  ) for i in _ind])
        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images, masks,rgb_masks, rgb_mask_mos, rgb_mos = zip(*arg_labels_and_images)
        return images, masks,rgb_masks, rgb_mask_mos, rgb_mos, labels

