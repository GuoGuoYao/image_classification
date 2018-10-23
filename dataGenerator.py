#-*-coding:utf-8-*-
from __future__ import print_function
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cv2
from random import shuffle
import os

class DataGenerator(object):
    def __init__(self,patch_size,batch_size,classnames,train_dir,ratio):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.datas = []
        self.labels = []

        # read the names of images
        csv = open(os.path.join(train_dir,'labels.csv'))
        for linea in csv.readlines():
            linea = linea.strip().split(',')
            img = linea[0]
            label = linea[1]
            self.datas.append(img)
            self.labels.append(label)
        self.keys = list(range(len(self.datas)))

        # train_file = os.listdir(train_dir)
        # for f_train in train_file:
        #     for p in os.listdir(os.path.join(train_dir,f_train)):
        #         tmp_path = os.path.join(train_dir, f, p)
        #         self.train_data.append(tmp_path)
        # valid_file = os.listdir(valid_dir)
        # for f_valid in valid_file:
        #     for p in os.listdir(os.path.join(valid_dir,f_valid)):
        #         tmp_path = os.path.join(valid_dir, f, p)
        #         self.valid_data.append(tmp_path)


        all_data = []
        all_label = []
        keys = self.keys
        shuffle(keys)
        total_num = len(self.datas)
        for d in range(total_num):
            all_data.append(self.datas[keys[d]])
            all_label.append(self.labels[keys[d]])
        self.train_datas = []
        self.train_labels = []
        self.valid_datas = []
        self.valid_labels = []
        self.train_datas = all_data[:int(total_num*ratio)]
        self.train_labels = all_label[:int(total_num*ratio)]
        self.train_batches = len(self.train_datas) // self.batch_size
        self.train_keys  = list(range(len(self.train_datas)))

        self.valid_datas = all_data[int(total_num * ratio):]
        self.valid_labels = all_label[:int(total_num*ratio)]
        self.valid_batches = len(self.valid_datas) // self.batch_size
        self.valid_keys = list(range(len(self.valid_datas)))




    def generator(self, trainflag):
        labelenconder = LabelEncoder()
        labelenconder.fit(self.classnames)
        if trainflag == True:
            img_datas = self.train_datas
            img_labels = self.train_labels
            sec_key = self.train_keys
            num_batch = self.train_batches
        else:
            img_datas = self.valid_datas
            img_labels = self.valid_labels
            sec_key = self.valid_keys
            num_batch = self.valid_batches
        while True:
            shuffle(sec_key)
            for i in range(num_batch):
                inputs = []
                targets = []
                for j in range(self.batch_size):
                    tmp_imgpath = img_datas[sec_key[j + i * self.batch_size]]
                    path_workplace = os.getcwd()
                    tmp_img = cv2.imread(os.path.join(path_workplace, 'train/imgs/',tmp_imgpath)).astype("float32")
                    lab = img_labels[sec_key[j + i * self.batch_size]]
                    img = cv2.resize(tmp_img, self.patch_size)
                    img = img * 1.0 / 255
                    inputs.append(img)
                    targets.append(lab)
                    data_inputs = np.array(inputs)
                #data_targets = np.array(data_targets).flatten()
                data_targets = labelenconder.transform(targets)
                data_targets = to_categorical(data_targets, self.num_classes)
                yield (data_inputs, data_targets)





