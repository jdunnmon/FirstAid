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
import random
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
# cropping_style = ['default', 'random', 'center']
def save_as_h5(data_path, root_path, folder_path, image_path, mask_path, label_dictionary, valid_filenames, cropping_style='default'):
    write_path_prefix = join(root_path,folder_path)
    #data_path_prefix = join(data_path, folder_path)
    if not isdir(write_path_prefix):
        mkdir(write_path_prefix)
    for fname in valid_filenames:
        #segment the filename
        patient_name = fname.split("_")[1]
        directory = join(write_path_prefix,'P_'+patient_name)
        if not isdir(directory):
            mkdir(directory)
        #check if directory exists, makedir if it doesnt
        #open image
        cur_image_path = join(join(data_path, image_path), fname)

        cur_mask_path = join(join(data_path, mask_path), fname)
        im = Image.open(cur_image_path)
        # if cropping_style == 'default':
        #     im = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)))
        # elif cropping_style == 'center':
        #     pre_crop_size = IMAGE_SIZE*3
        #     im = np.array(im.resize((pre_crop_size, pre_crop_size)))
        #     im = im[224:448,224:448]
        # elif cropping_style == 'random':
        #     pre_crop_size = IMAGE_SIZE*3
        #     im = np.array(im.resize((pre_crop_size, pre_crop_size)))
        #     start_indices = [0, 224, 448]
        #     x_start = random.choice(start_indices)
        #     y_start = random.choice(start_indices)
        #     im = im[y_start:y_start+224,x_start:x_start+224]
        im = np.array(im)
        im = img_as_float(im)
        im = im[:,:,np.newaxis]
        #create data
        #open mask
        mask = Image.open(cur_mask_path)
        mask = np.array(mask)
        mask = mask.astype(np.uint64)

        #get label
        label = label_dictionary[fname]
        #save the h5 image in the right patient subfolder
        fname_stripped = fname[:len(fname)-4]
        path_h5 = join(directory, fname_stripped+'.h5')
        print "H5 FILE PATH: ", path_h5
        h5f = h5py.File(path_h5, 'w')
        h5f.create_dataset('data', data=im)
        h5f.create_dataset('seg', data=mask)
        #h5f.create_dataset('seg', data=label)
        h5f.create_dataset('label', data=label)
        h5f.close()



