from __future__ import print_function  # Only needed for Python 2
import numpy as np
from collections import defaultdict
import os


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def create_log_dict(filename):
    log_dict = defaultdict(list)

    with open(filename, 'r') as file_out:

        for ii,line in enumerate(file_out.readlines()):
            words = line.split()
            if words[0] == 'Iter:':
                for jj,word in enumerate(words):
                    if not is_number(word):
                        log_dict[word[:-1]].append(float(words[jj+1]))

            elif len(words) <8 and ii != 0: 
                col_split = line.split(':')
                log_dict[col_split[0]].append(float(col_split[1]))
            
    return log_dict

def get_log_paths(root):
    lst = []
    names = []
    for subdir, dirs, files in os.walk(root):
        for fil in files:
            if "internal" in fil:
                lst.append(os.path.join(subdir,fil))
                namesplit = os.path.split(subdir)
                names.append(namesplit[1])    #getting experiment name
    exp_name = os.path.split(namesplit[0])[1]
    return lst, names, exp_name, 


def create_master_dict(log_paths,run_names):
    all_log_dict = defaultdict(list)
    for ii,log in enumerate(log_paths):
        all_log_dict[run_names[ii]] = create_log_dict(log)
    return all_log_dict

def find_max_by_param(all_dict,param):
    """
    INPUTS
    param: parameter in dictionary to maximize over hyperparameters
    
    OUTPUTS
    key: key to dict containing values from run with max value of param
    
    """
    max_val = 0
    for ii,ky in enumerate(all_dict.keys()):
        if all_dict[ky][param] != []:
            max_temp = max(all_dict[ky][param])
        else:
            max_temp = -1
        if max_temp>max_val:
            max_val = max_temp
            key = ky

    return key, max_val


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def parse_param_vals(log,exp_name):
    params = {}
    param_lst = remove_prefix(log,exp_name+'_')
    words = param_lst.split('_')
    for ii,word in enumerate(words):
        if ii % 2:
            params[words[ii-1]] = words[ii]
    return params

def print_dict(dct,f):
    statement = str()
    for ky in dct.keys():
        print(str(ky)+': '+str(dct[ky]),file=f)
