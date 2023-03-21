import cv2
import os
import random
import numpy as np
from mindspore import Tensor

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

class MyDataSet():
    def __init__(self, root, list_name, train_num, image_size,dataset):
        self.root = root
        self.list_path = self.root + list_name
        self.h, self.w = image_size
        self.dataset = dataset
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if train_num>700:
            self.img_ids = self.img_ids[:]
        else:
            self.img_ids = self.img_ids[:train_num]
        self.files = []
        for name in self.img_ids:
            if self.dataset == 'skin':
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            else:
                img_file = os.path.join(self.root, "images/%s.png" % name)
            label_file = os.path.join(self.root, "masks/%s.png" % name)
            self.files.append({"img": img_file,"label": label_file, "name": name})
        #np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
            
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = Tensor.from_numpy(image.astype(np.float32))
        label = Tensor.from_numpy(label.astype(np.uint8)).long()
        return image, label
