import json
import pandas as pd
import os
from os.path import join
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import shutil

class DDSMDataOfficial:
    # split_option = ['MLO', 'CC', 'ALL']
    def __init__(self,root_path,write_path,label_json,image_path, test_image_path, mask_path='.', test_mask_path='.',split_option='ALL',n_splits=1,split_seed='None'):
        self.split_option = split_option
        self.write_path = write_path
        self.root_path = root_path
        self.label_json = label_json
        self.image_path = image_path
        self.test_image_path = test_image_path
        self.mask_path = mask_path
        self.test_mask_path = test_mask_path
        self.img_names = self.load_image_names()
        self.msk_names = self.load_mask_names()
        self.val_names = self.get_valid_labels()
        self.test_img_names = self.load_test_image_names()
        self.test_msk_names = self.load_test_mask_names()
        self.test_val_names = self.get_test_valid_labels()

    def load_json_file(self,path):
        data = open(path, 'r').read()
        try:
            return json.loads(data)
        except ValueError, e:
            raise MalformedJsonFileError('%s when reading "%s"' % (str(e),path))

    def load_labels(self):
        f = self.load_json_file(join(self.root_path,self.label_json))
        return f

    def load_file_names(self,mypath):
        f = []
        for (dirpath, dirnames, filenames) in os.walk(mypath):
            # ANN: experimenting
            #print filenames
            f.extend(filenames)
            break
        return f

    def load_image_names(self):
        f = self.load_file_names(join(self.root_path,self.image_path))
        return f

    def load_test_image_names(self):
        f = self.load_file_names(join(self.root_path,self.test_image_path))
        return f

    def load_mask_names(self):
        f = self.load_file_names(join(self.root_path,self.mask_path))
        return f

    def load_test_mask_names(self):
        f = self.load_file_names(join(self.root_path,self.test_mask_path))
        return f

    def get_valid_labels(self):
        if self.split_option == 'MLO':
            f = [a for a in self.img_names if a in self.msk_names and 'MLO' in a]
            return f
        if self.split_option == 'CC':
            f = [a for a in self.img_names if a in self.msk_names and 'CC' in a]
            return f
        f = [a for a in self.img_names if a in self.msk_names]
        return f

    def get_test_valid_labels(self):
        if self.split_option == 'MLO':
            f = [a for a in self.test_img_names if a in self.test_msk_names and 'MLO' in a]
            return f
        if self.split_option == 'CC':
            f = [a for a in self.test_img_names if a in self.test_msk_names and 'CC' in a]
            return f
        f = [a for a in self.test_img_names if a in self.test_msk_names]
        return f

    def create_train_splits(self, n_splits, train_size=0.85, val_size=0.15, rand_seed=None):
        sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size,\
                                               train_size=train_size, random_state=rand_seed)
        lab_dict = self.load_labels()
        imgs = np.array(self.val_names)
        if imgs[0] == '.DS_Store':
            imgs = imgs[1:]
        labs = np.array([lab_dict[a] for a in imgs])

        trainidx = []
        valididx = []
        for trnvalx,valx in sss_train_val.split(imgs,labs):
            trainidx.append(trnvalx)
            valididx.append(valx)
        x_train = []
        x_val = []
        y_train = []
        y_val = []

        for ii in range(len(trainidx)):
            x_train.append(imgs[trainidx[ii]])
            y_train.append(labs[trainidx[ii]])

        for ii in range(len(valididx)):
            x_val.append(imgs[valididx[ii]])
            y_val.append(labs[valididx[ii]])

        return x_train[0], y_train[0], x_val[0], y_val[0]

    def create_test(self):
        lab_dict = self.load_labels()
        imgs = np.array(self.test_val_names)
        if imgs[0] == '.DS_Store':
            imgs = imgs[1:]
        labs = np.array([lab_dict[a] for a in imgs])
        y_test = np.array([lab_dict[a] for a in imgs])
        x_test = imgs
        return x_test, y_test

    def copy_file(self,source,dest):
        try:
            shutil.copyfile(source, dest)
        except ValueError:
            print("Error: File does not exist!")

    def make_new_direc(self,name):
        if os.path.exists(name):
            shutil.rmtree(name)
            os.makedirs(name)
        else:
            os.makedirs(name)

    def write_split_stats(self, train_split, val_split, train_len, val_len, test_len, log_dir):
        # write stats
        # split_stats
        # train/test/val class balances and sizes
        log_direc = join(self.root_path,log_dir)
        with open(join(log_direc,"split_stats.txt"), 'w') as f:
            f.write("train / val: " + str(train_split) + " / " + str(val_split) + "\n")
            f.write("train / val / test sizes: " + str(train_len) + " / " + str(val_len) + " / " + str(test_len))

    def write_splits(self,x_train,x_val,x_test,log_dir,suffix=""):
        """
        designed to emulate original file structure for ddsm tanda experiments -- can be improved
        """
        #defining and creating directory structure
        log_direc = join(self.write_path,log_dir)
        train_direc = join(self.write_path,'train_set'+suffix)
        val_direc = join(self.write_path,'val_set'+suffix)
        test_direc = join(self.write_path,'test_set'+suffix)

        self.make_new_direc(log_direc)
        self.make_new_direc(train_direc)
        self.make_new_direc(val_direc)
        self.make_new_direc(test_direc)

        #writing log files

        with open(join(log_direc,"train.txt"), "w") as outfile:
            for s in x_train:
                outfile.write("%s\n" % s)

        with open(join(log_direc,"val.txt"), "w") as outfile:
            for s in x_val:
                outfile.write("%s\n" % s)

        with open(join(log_direc,"test.txt"), "w") as outfile:
            for s in x_test:
                outfile.write("%s\n" % s)

        #copying files to directory structure
        if not os.path.exists(train_direc):
            os.makedirs(train_direc)
        for file_name in x_train:
            #print "ROOT PATH", self.root_path
            #print "IMAGE PATH", self.image_path
            #print "FILE NAME", file_name
            full_file_name = join(self.root_path, self.image_path, file_name)
            new_file = join(train_direc,file_name)
            self.copy_file(full_file_name,new_file)

        for file_name in x_val:
            full_file_name = join(self.root_path, self.image_path, file_name)
            new_file = join(val_direc,file_name)
            self.copy_file(full_file_name,new_file)

        for file_name in x_test:
            full_file_name = join(self.root_path, self.test_image_path, file_name)
            new_file = join(test_direc,file_name)
            self.copy_file(full_file_name,new_file)

    def create_and_write_splits(self, n_splits, log_dir, suffix="", train_size=0.85, val_size=0.15, rand_seed=None):
        x_train, y_train, x_val, y_val = self.create_train_splits(n_splits, train_size, val_size, rand_seed)
        x_test, y_test = self.create_test()
        self.write_splits(x_train, x_val, x_test, log_dir, suffix)
        self.write_split_stats(train_size, val_size, len(x_train), len(x_val), len(x_test), log_dir)