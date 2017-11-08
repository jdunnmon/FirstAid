import numpy as np
import h5py
from PIL import Image
from skimage import img_as_float
import json
import pandas as pd
import os
from os.path import join
import shutil
from os import listdir,mkdir
from os.path import isdir,join
IMAGE_SIZE = 224

def load_json_file(path):
    data = open(path, 'r').read()
    try:
        return json.loads(data)
    except ValueError, e:
        raise MalformedJsonFileError('%s when reading "%s"' % (str(e),path))

def load_labels(root_path, label_json):
    f = load_json_file(join(root_path,label_json))
    return f

def load_file_names(mypath):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        # ANN: experimenting
        #print filenames
        f.extend(filenames)
        break
    return f

def load_image_names(root_path, img_path):
    f = load_file_names(join(root_path,img_path))
    return f

def get_valid_files(img_filenames, mask_filenames):
    f = [a for a in img_filenames if a in mask_filenames]
    return f


# folder_path defines whether it is train, test, or validation, i.e. the name
def save_as_h5(root_path, folder_path, image_path, mask_path, label_dictionary, valid_filenames):
    path_prefix = join(root_path,folder_path)
    if not isdir(path_prefix):
        mkdir(path_prefix)
    for fname in valid_filenames:
        #segment the filename
        patient_name = fname.split("_")[1]
        directory = join(path_prefix,'P_'+patient_name)
        if not isdir(directory):
            mkdir(directory)
        #check if directory exists, makedir if it doesnt
        #open image
        cur_image_path = join(join(root_path, image_path), fname)

        #cur_mask_path = join(join(root_path, mask_path), fname)
        im = Image.open(cur_image_path)
        im = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)))
        im = img_as_float(im)
        im = im[:,:,np.newaxis]
        #create data
        #open mask
        #mask = Image.open(mask_path)
        #mask = np.array(mask.resize((IMAGE_SIZE, IMAGE_SIZE)))
        #mask = mask.astype(np.int64)

        #get label
        label = label_dictionary[fname]
        #save the h5 image in the right patient subfolder
        fname_stripped = fname[:len(fname)-4]
        path_h5 = join(directory, fname_stripped+'.h5')
        h5f = h5py.File(path_h5, 'w')
        h5f.create_dataset('data', data=im)
        #h5f.create_dataset('seg', data=label)
        h5f.create_dataset('label', data=label)

root_path = '/Users/annhe/Projects/tandaExperiment/ddsm-data-official/'
train_image_path = 'train_set_mlo'
val_image_path = 'val_set_mlo'
test_image_path = 'test_set_mlo'
train_mask_path = 'train_masks_cropped'
val_mask_path = 'train_masks_cropped'
test_mask_path = 'test_masks_cropped'
label_json = 'mass_to_label.json'
label_dictionary = load_labels(root_path, label_json)
train_valid_filenames = get_valid_files(load_image_names(root_path, train_image_path), load_image_names(root_path, train_mask_path))
save_as_h5(root_path, 'h5_train_set_mlo', train_image_path, train_mask_path, label_dictionary, train_valid_filenames)
test_valid_filenames = get_valid_files(load_image_names(root_path, test_image_path), load_image_names(root_path, test_mask_path))
save_as_h5(root_path, 'h5_test_set_mlo', test_image_path, test_mask_path, label_dictionary, test_valid_filenames)
val_valid_filenames = get_valid_files(load_image_names(root_path, val_image_path), load_image_names(root_path, val_mask_path))
save_as_h5(root_path, 'h5_val_set_mlo', val_image_path, val_mask_path, label_dictionary, val_valid_filenames)



