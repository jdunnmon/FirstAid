from save_as_h5 import *

root_path = '/Volumes/ANN_HE/ddsm-processed/'
train_image_path = 'train_set_full'
val_image_path = 'val_set_full'
test_image_path = 'test_set_full'
train_mask_path = 'train_masks'
val_mask_path = 'train_masks'
test_mask_path = 'test_masks'
label_json = 'mass_to_label.json'
label_dictionary = load_labels(root_path, label_json)

train_valid_filenames = get_valid_files(load_image_names(root_path, train_image_path), load_image_names(root_path, train_mask_path))
save_as_h5(root_path, 'h5_train_set', train_image_path, train_mask_path, label_dictionary, train_valid_filenames, cropping_style='default')

test_valid_filenames = get_valid_files(load_image_names(root_path, test_image_path), load_image_names(root_path, test_mask_path))
save_as_h5(root_path, 'h5_test_set', test_image_path, test_mask_path, label_dictionary, test_valid_filenames, cropping_style='default')

val_valid_filenames = get_valid_files(load_image_names(root_path, val_image_path), load_image_names(root_path, val_mask_path))
save_as_h5(root_path, 'h5_val_set', val_image_path, val_mask_path, label_dictionary, val_valid_filenames, cropping_style='default')