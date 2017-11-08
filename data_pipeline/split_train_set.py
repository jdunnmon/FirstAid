import numpy as np
import json, os
import matplotlib as plt
from data_utils_train_test import DDSMDataOfficial

root_path = '/Users/annhe/Projects/tandaExperiment/ddsm-data-official/'
label_json = 'mass_to_label.json'
image_path = 'train_mammo_cropped'
mask_path = 'train_masks_cropped'
test_image_path = 'test_mammo_cropped'
test_mask_path = 'test_masks_cropped'
rand_seed = 1
data = DDSMDataOfficial(root_path,label_json,image_path,test_image_path, mask_path=mask_path,test_mask_path=test_mask_path, mlo_only=True)
x_train,y_train,x_val,y_val= \
data.create_train_splits(1,train_size=0.85,val_size=0.15,rand_seed=rand_seed)
x_test,y_test = data.create_test()
data.write_splits(x_train,x_val,x_test,'mlo_split_0',suffix="_mlo")
print len(x_train),len(y_train),len(x_val),len(y_val),len(x_test),len(y_test)