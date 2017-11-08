from save_as_h5 import *

root_path = '/Users/annhe/Projects/tandaExperiment/FirstAid/data_pipeline/'
train_image_path = 'train_set_mlo'
val_image_path = 'val_set_mlo'
test_image_path = 'test_set_mlo'
train_mask_path = 'train_masks_cropped'
val_mask_path = 'train_masks_cropped'
test_mask_path = 'test_masks_cropped'
label_json = 'mass_to_label.json'
label_dictionary = load_labels(root_path, label_json)

train_valid_filenames = get_valid_files(load_image_names(root_path, train_image_path), load_image_names(root_path, train_mask_path))
save_as_h5(root_path, 'h5_train_set_mlo', train_image_path, train_mask_path, label_dictionary, train_valid_filenames, cropping_style='random')

test_valid_filenames = get_valid_files(load_image_names(root_path, test_image_path), load_image_names(root_path, test_mask_path))
save_as_h5(root_path, 'h5_test_set_mlo', test_image_path, test_mask_path, label_dictionary, test_valid_filenames, cropping_style='random')

val_valid_filenames = get_valid_files(load_image_names(root_path, val_image_path), load_image_names(root_path, val_mask_path))
save_as_h5(root_path, 'h5_val_set_mlo', val_image_path, val_mask_path, label_dictionary, val_valid_filenames, cropping_style='random')