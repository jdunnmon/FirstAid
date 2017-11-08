import numpy as np
import json, os
import matplotlib as plt
from data_utils_train_test import DDSMDataOfficial

root_path = '/Users/annhe/Projects/tandaExperiment/ddsm-data-official/'
write_path = '/Users/annhe/Projects/tandaExperiment/FirstAid/data_pipeline/'
label_json = 'mass_to_label.json'
image_path = 'train_mammo_cropped'
mask_path = 'train_masks_cropped'
test_image_path = 'test_mammo_cropped'
test_mask_path = 'test_masks_cropped'
rand_seed = 1
data = DDSMDataOfficial(root_path,write_path,label_json,image_path,test_image_path, mask_path=mask_path,test_mask_path=test_mask_path, split_option="MLO")
data.create_and_write_splits(1,'mlo_split_0',suffix="_mlo",train_size=0.85,val_size=0.15,rand_seed=rand_seed)