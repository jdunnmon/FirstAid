from __future__ import print_function  # Only needed for Python 2
import numpy as np
import os

from analysis_utils import *

#setting the root log directory
root_dir = '/lfs/local/0/jdunnmon/data_aug/firstaid/all_runs/logs/11_14_17/hp_search_GoogLe'

#setting output file
outfile = os.path.join(root_dir,'hp_search_results.txt')

#getting log paths and hp search run names
log_paths, log_names, exp_name = get_log_paths(root_dir)

#creating dict of dicts keyed by log parameters
all_log_dict = create_master_dict(log_paths,log_names)

#searching for maximum test accuracy
max_testacc_trial, max_testacc = find_max_by_param(all_log_dict,'Test Accuracy')

#searching for maximum validation accuracy
max_valacc_trial, max_valacc  = find_max_by_param(all_log_dict,'Validation Accuracy')

#searching for maximum train accuracy
max_trainacc_trial, max_trainacc = find_max_by_param(all_log_dict,'Train Accuracy')

#writing best hps to output file
f = open(outfile, 'w')

print("Best Test Accuracy = %.2f" % max_testacc[0],file=f)
print("Best Parameters for Test Accuracy:",file=f)
print_dict(parse_param_vals(max_testacc_trial, exp_name),f=f)
print("\n")

print("Best Val Accuracy = %.2f" % max_valacc[0],file=f)
print("Best Parameters for Val Accuracy:",file=f)
print_dict(parse_param_vals(max_valacc_trial, exp_name),f=f)
print("\n")

print("Best Train Accuracy = %.2f" % max_trainacc[0],file=f)
print("Best Parameters for Train Accuracy:",file=f)
print_dict(parse_param_vals(max_trainacc_trial,exp_name),f=f)

f.close()
