import numpy as np
import json, os
import matplotlib as plt
from data_utils_train_test import DDSMDataOfficial

root_path = '/Volumes/ANN_HE/ddsm-processed/'
write_path = '/Volumes/ANN_HE/ddsm-processed/'
label_json = 'mass_to_label.json'
image_path = 'train_mammo'
mask_path = 'train_masks'
test_image_path = 'test_mammo'
test_mask_path = 'test_masks'
rand_seed = 1
data = DDSMDataOfficial(root_path,write_path,label_json,image_path,test_image_path, mask_path=mask_path,test_mask_path=test_mask_path, split_option="ALL")
data.create_and_write_splits(1,'full_split_0',suffix="_full",train_size=0.85,val_size=0.15,rand_seed=rand_seed)