#!/bin/bash
# Save dicom files as tiff
declare -r root_path="/Users/annhe/Projects/tandaExperiment/ddsm-data-official/"
declare -r train_csv="/train/mass_case_description_train_set.csv"
declare -r train_image_dir="/train/DOI/"
declare -r test_csv="/test/mass_case_description_test_set.csv"
declare -r test_image_dir="/test/DOI/"
declare -r write_directory="/Users/annhe/Projects/tandaExperiment/FirstAid/data_pipeline/"

export MATLABPATH=/Users/annhe/Projects/tandaExperiment/tanda/experiments/ddsm/data_pipeline
/Applications/MATLAB_R2016a.app/bin/matlab -nodisplay -nodesktop -r "process_dicom('$root_path','$train_image_dir','$train_csv','$test_image_dir','$test_csv','$write_directory'); quit();"
